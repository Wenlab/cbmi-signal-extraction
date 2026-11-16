"""Compare independently fitted two-camera transforms with saved calibration.

Run with ``pixi run python scripts/compare_calibration_20261104.py``.
Raw TIFFs are single (Y, X) frames. A 20-frame block reproduces the project's
MIP/ECC calculation without asserting an unverified frame-to-Z mapping.
"""

from __future__ import annotations

import base64
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import tifffile


WORKSPACE = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"H:\Process_temporary\cali_2026-11-04_15-10-16")
OUTPUT = WORKSPACE / "output" / "calibration_comparison_20261104"
SEED = 72
NUM_FIT_BLOCKS = 10
SIFT_RATIO = 0.7
RANSAC_THRESHOLD_PX = 2.0


def log(message):
    print(message, flush=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def homogeneous(matrix):
    return np.vstack([np.asarray(matrix, dtype=np.float64), [0, 0, 1]])


def flipped_convention(matrix, width):
    """Convert flipped-red -> green to red -> flipped-green, or vice versa."""
    flip = np.array([[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]], dtype=float)
    return (flip @ homogeneous(matrix) @ flip)[:2]


def parameters(matrix):
    # Match OpenCV getRotationMatrix2D and the JSON's negative angle convention.
    return {
        "scale": float(np.hypot(matrix[0, 0], matrix[1, 0])),
        "rotation_deg": float(-np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
        "tx": float(matrix[0, 2]),
        "ty": float(matrix[1, 2]),
    }


def median_similarity(matrices):
    values = np.array([list(parameters(m).values()) for m in matrices])
    scale, rotation_deg, tx, ty = np.median(values, axis=0)
    angle = -np.radians(rotation_deg)
    a, b = scale * np.cos(angle), scale * np.sin(angle)
    return np.array([[a, -b, tx], [b, a, ty]])


def apply_points(matrix, points):
    return points @ matrix[:, :2].T + matrix[:, 2]


def displacement_summary(matrix, original, shape):
    """Compare predictions on every pixel center, using zero-based (x, y)."""
    height, width = shape
    delta = np.asarray(matrix) - np.asarray(original)
    y, x = np.mgrid[:height, :width]
    dx = delta[0, 0] * x + delta[0, 1] * y + delta[0, 2]
    dy = delta[1, 0] * x + delta[1, 1] * y + delta[1, 2]
    distance = np.hypot(dx, dy)
    center = np.array([[(width - 1) / 2, (height - 1) / 2]])
    corners = np.array([[0, 0], [width - 1, 0], [0, height - 1],
                        [width - 1, height - 1]])
    return {
        "rms_px": float(np.sqrt(np.mean(distance**2))),
        "mean_px": float(distance.mean()),
        "median_px": float(np.median(distance)),
        "p95_px": float(np.percentile(distance, 95)),
        "max_px": float(distance.max()),
        "center_displacement_xy_px": apply_points(delta, center)[0].tolist(),
        "corner_displacements_xy_px": apply_points(delta, corners).tolist(),
    }


def uint8_image(image):
    low, high = np.percentile(image, [1, 99])
    return np.uint8(np.clip((image.astype(np.float32) - low) / (high - low), 0, 1) * 255)


def feature_fit(red_flipped, green):
    """Mutual descriptor matching avoids repeated-bar false correspondences."""
    detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    source_keys, source_desc = detector.detectAndCompute(uint8_image(red_flipped), None)
    target_keys, target_desc = detector.detectAndCompute(uint8_image(green), None)
    if source_desc is None or target_desc is None:
        raise ValueError("SIFT found no descriptors")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(source_desc, target_desc, k=2)
    reverse = matcher.knnMatch(target_desc, source_desc, k=2)
    reverse_good = {
        first.queryIdx: first.trainIdx
        for pair in reverse if len(pair) == 2
        for first, second in [pair] if first.distance < SIFT_RATIO * second.distance
    }
    matches = [
        first for pair in forward if len(pair) == 2 for first, second in [pair]
        if first.distance < SIFT_RATIO * second.distance
        and reverse_good.get(first.trainIdx) == first.queryIdx
    ]
    if len(matches) < 25:
        raise ValueError(f"Only {len(matches)} mutual matches")
    source = np.float32([source_keys[m.queryIdx].pt for m in matches])
    target = np.float32([target_keys[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(SEED)
    matrix, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_PX,
        maxIters=10000, confidence=0.999, refineIters=20,
    )
    if matrix is None or inliers is None or int(inliers.sum()) < 25:
        raise ValueError("Not enough geometrically consistent feature matches")
    keep = inliers.ravel().astype(bool)
    return matrix, source[keep].astype(float), target[keep].astype(float), {
        "red_keypoints": len(source_keys),
        "green_keypoints": len(target_keys),
        "mutual_ratio_matches": len(matches),
        "num_inliers": int(keep.sum()),
        "inlier_rmse_px": float(np.sqrt(np.mean(
            np.sum((apply_points(matrix, source[keep]) - target[keep])**2, axis=1)
        ))),
    }


def correlation(first, second, mask):
    a = first[mask].astype(np.float64)
    b = second[mask].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def evaluate_images(red_flipped, green, matrices):
    height, width = green.shape
    # A single common support prevents different border crops favoring a method.
    common = np.ones(green.shape, dtype=bool)
    ones = np.ones(green.shape, dtype=np.uint8)
    for matrix in matrices.values():
        common &= cv2.warpAffine(
            ones, matrix, (width, height), flags=cv2.INTER_NEAREST
        ).astype(bool)
    common = cv2.erode(common.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    smooth_red = cv2.GaussianBlur(red_flipped.astype(np.float32), (0, 0), 1.5)
    smooth_green = cv2.GaussianBlur(green.astype(np.float32), (0, 0), 1.5)
    scores = {}
    for name, matrix in matrices.items():
        warped = cv2.warpAffine(smooth_red, matrix, (width, height))
        scores[name] = correlation(warped, smooth_green, common)
    scores["flip_only"] = correlation(smooth_red, smooth_green, common)
    return scores


def camera_metadata(folder, files):
    with (folder / "metadata.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ticks = np.array([int(r["tick_count"]) / int(r["tick_frequency_Hz"]) for r in rows])
    ids = [int(r["save_frame_number"]) for r in rows]
    if ids != list(range(len(files))) or [int(p.stem) for p in files] != ids:
        raise ValueError(f"Missing or mismatched TIFF/metadata IDs: {folder}")
    return {
        "num_frames": len(files),
        "median_frame_interval_s": float(np.median(np.diff(ticks))),
        "first_metadata": rows[0],
        "last_metadata": rows[-1],
        "metadata_sha256": digest(folder / "metadata.csv"),
        "camera_info_sha256": digest(folder / "camera_info.csv"),
        "camera_info": (folder / "camera_info.csv").read_text(),
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def overlay(red_flipped, green, matrix):
    """Dark target -> bright red/green markings on a dark display."""
    height, width = green.shape
    red = cv2.warpAffine(255 - uint8_image(red_flipped), matrix, (width, height))
    green_display = 255 - uint8_image(green)
    return np.stack([np.zeros_like(red), green_display, red], axis=-1)


def make_overview(red_flipped, green, matrices, original, path):
    tiles = []
    names = ["flip_only", "calib_json", "sift_similarity", "pipeline_ecc"]
    titles = ["Horizontal flip only", "Existing calib_params.json",
              "Independent SIFT similarity", "Repository MIP / rigid ECC"]
    for name, title in zip(names, titles):
        matrix = matrices.get(name, np.eye(2, 3))
        im = cv2.resize(overlay(red_flipped, green, matrix), (512, 512))
        tile = np.full((568, 512, 3), 245, dtype=np.uint8)
        tile[56:] = im
        cv2.putText(tile, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.57, (30, 30, 30), 1, cv2.LINE_AA)
        difference = displacement_summary(matrix, original, green.shape)
        label = f"vs JSON: RMS {difference['rms_px']:.3f} px"
        cv2.putText(tile, label, (12, 46), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (65, 65, 65), 1, cv2.LINE_AA)
        tiles.append(tile)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))


def encoded_png(image):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def write_html(report, red_flipped, green, matrices):
    payload = {
        "red": encoded_png(255 - uint8_image(red_flipped)),
        "green": encoded_png(255 - uint8_image(green)),
        "matrices": {name: m.tolist() for name, m in matrices.items()},
        "differences": report["comparison_to_json"],
        "scores": report["validation"]["summary"],
    }
    rows = ""
    for name, matrix in matrices.items():
        p = parameters(matrix)
        d = report["comparison_to_json"][name]
        s = report["validation"]["summary"][name]
        rows += (
            f"<tr><td>{name}</td><td>{p['scale']:.9f}</td>"
            f"<td>{p['rotation_deg']:.6f}</td><td>{p['tx']:.6f}</td>"
            f"<td>{p['ty']:.6f}</td><td>{d['rms_px']:.4f}</td>"
            f"<td>{d['max_px']:.4f}</td><td>{s['mean_frame_feature_rmse_px']:.4f}</td>"
            f"<td>{s['mean_ncc']:.6f}</td></tr>"
        )
    template = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>红绿通道标定矩阵比较</title>
<style>
body{margin:28px auto;max-width:1180px;padding:0 22px;font:15px/1.7 system-ui;color:#1d2736;background:#f5f7fa}
h1{font-size:27px}h2{font-size:19px}section{background:white;border:1px solid #dae0e8;border-radius:10px;padding:20px;margin:18px 0}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;text-align:right;border-bottom:1px solid #e0e4ea}td:first-child,th:first-child{text-align:left}
canvas{max-width:100%;height:auto;background:#000;border:1px solid #aaa}
.controls{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin-bottom:16px}
select,button{padding:7px}input{vertical-align:middle}code{background:#edf1f6;padding:2px 4px}
pre{overflow:auto;background:#f0f3f8;padding:12px}.small{color:#536274;font-size:13px}
</style>
<h1>红绿通道标定矩阵比较</h1>
<p>原始数据：<code>H:\Process_temporary\cali_2026-11-04_15-10-16</code>。254 对 1024 × 1024 单帧 TIFF；所有计算均使用原始 uint16 图像。</p>
<section><h2>相同变换模型的比较</h2>
<p>所有表中矩阵统一为 <b>水平翻转后的红通道 → 原始绿通道</b>。像素中心坐标为零起点 (x, y)，翻转为 x′ = 1023 − x。角度采用 JSON / OpenCV 符号约定。</p>
<table><thead><tr><th>方法</th><th>缩放</th><th>角度 (°)</th><th>tx (px)</th><th>ty (px)</th><th>对 JSON<br>全场 RMS (px)</th><th>最大差异 (px)</th><th>留出帧<br>特征 RMSE (px)</th><th>留出帧 NCC</th></tr></thead><tbody>__ROWS__</tbody></table>
<p class="small">SIFT similarity 使用每个拟合块中一张原始帧：双向 SIFT ratio 0.7、RANSAC 2 px、10 个逐帧相似变换的参数中位数。pipeline_ecc 直接调用当前仓库的 10 块 MIP / rigid ECC；它不估计缩放。54 张留出帧没有参与任何最终矩阵的拟合。特征 RMSE 为每帧误差的算术平均，NCC 为共同有效区域上、sigma=1.5 px 高斯平滑后的 Pearson 相关。</p>
</section>
<section><h2>交互检查</h2>
<div class="controls"><label>矩阵 <select id="method"></select></label>
<label>显示 <select id="mode"><option value="overlay">红绿叠加</option><option value="difference">亮度归一化差值</option><option value="blink">红 / 绿闪烁</option></select></label>
<label>亮度 <input type="range" id="gain" min="0.5" max="4" step="0.1" value="1.5"></label></div>
<canvas id="view" width="1024" height="1024"></canvas>
<p>红色 = 变换后的红通道；绿色 = 原始绿通道；黄色 = 两者重合。两通道分别用 1–99 百分位调整显示，显示亮度不用于配准。示例帧：__EXAMPLE__。</p>
<pre id="matrix"></pre>
</section>
<section><h2>矩阵差异的空间分布</h2><canvas id="field" width="1024" height="550"></canvas>
<p class="small">箭头为当前矩阵相对 JSON 的位移差，放大 100 倍；背景为绿通道的缩略图。图像显示坐标与位移评价坐标一致。标尺箭头表示真实 0.1 px。</p></section>
<section><h2>数据与结论的边界</h2>
<p>采集设置的扫描周期为 20 帧。当前仓库以 20 帧非重叠块做 MIP，12 个完整块中固定种子 72 选 10 块拟合；剩下两块和末尾 14 帧用于验证。没有明确的逐帧 Z 坐标、z-step 或样品平面像素尺寸，因此报告仅比较二维几何并以像素为单位，不将单帧当作神经活动时间点，也不确认一个块就是一个完整三维体积。</p>
<p>相似模型与刚体模型自由度不同；刚体 ECC 与 JSON 的差异包含固定缩放造成的折衷。连续帧来自同一标定板、同次记录，属于技术重复；留出验证不代表跨动物或跨实验独立复现。RANSAC 留出特征对应关系是图像估计，不是人工标定真值；NCC 受通道光照、离焦和噪声影响，不能单独证明几何正确。</p>
<p>软件版本、输入清单、源文件哈希、抽样帧号、逐帧结果、坐标转换和全部矩阵见同目录 JSON / CSV / NPY 文件。原始文件未写入。</p>
</section>
<script>
const D=__PAYLOAD__;
const selector=document.getElementById('method'), mode=document.getElementById('mode');
const gain=document.getElementById('gain'),canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
const field=document.getElementById('field'),fc=field.getContext('2d');
for(const key of Object.keys(D.matrices)){let o=document.createElement('option');o.value=o.textContent=key;selector.append(o)}
selector.value='sift_similarity';
const red=new Image(),green=new Image();let loaded=0,blink=false;
red.onload=green.onload=()=>{if(++loaded===2)draw()};red.src=D.red;green.src=D.green;
const source=document.createElement('canvas');source.width=source.height=1024;
const sc=source.getContext('2d',{willReadFrequently:true});
const target=document.createElement('canvas');target.width=target.height=1024;
const tc=target.getContext('2d',{willReadFrequently:true});
function draw(){
if(loaded!==2)return;
const m=D.matrices[selector.value];sc.setTransform(1,0,0,1,0,0);sc.clearRect(0,0,1024,1024);
sc.setTransform(m[0][0],m[1][0],m[0][1],m[1][1],m[0][2]+.5-.5*(m[0][0]+m[0][1]),m[1][2]+.5-.5*(m[1][0]+m[1][1]));sc.drawImage(red,0,0);
tc.clearRect(0,0,1024,1024);tc.drawImage(green,0,0);
const r=sc.getImageData(0,0,1024,1024),g=tc.getImageData(0,0,1024,1024),o=ctx.createImageData(1024,1024);
const v=Number(gain.value);
for(let i=0;i<o.data.length;i+=4){let a=r.data[i],b=g.data[i];
if(mode.value==='difference'){let t=(a-b)*v;o.data[i]=Math.max(0,t);o.data[i+1]=0;o.data[i+2]=Math.max(0,-t)}
else if(mode.value==='blink'){o.data[i]=o.data[i+1]=o.data[i+2]=(blink?a:b)*v}
else{o.data[i]=a*v;o.data[i+1]=b*v;o.data[i+2]=0}o.data[i+3]=255}
ctx.putImageData(o,0,0);
document.getElementById('matrix').textContent=selector.value+'\n'+JSON.stringify(m,null,2)+'\n全场 RMS 差异 = '+D.differences[selector.value].rms_px.toFixed(4)+' px';
drawField(m)}
function drawField(m){
fc.fillStyle='#111';fc.fillRect(0,0,1024,550);fc.globalAlpha=.45;fc.drawImage(green,0,0,550,550);fc.globalAlpha=1;
const ref=D.matrices.calib_json;fc.strokeStyle='#57ddff';fc.fillStyle='#57ddff';fc.lineWidth=1.7;
function arrow(x,y,u,v){fc.beginPath();fc.moveTo(x,y);fc.lineTo(x+u,y+v);fc.stroke();const a=Math.atan2(v,u);fc.beginPath();fc.moveTo(x+u,y+v);fc.lineTo(x+u-5*Math.cos(a-.45),y+v-5*Math.sin(a-.45));fc.lineTo(x+u-5*Math.cos(a+.45),y+v-5*Math.sin(a+.45));fc.closePath();fc.fill()}
for(let y=64;y<1024;y+=128)for(let x=64;x<1024;x+=128){const dx=(m[0][0]-ref[0][0])*x+(m[0][1]-ref[0][1])*y+m[0][2]-ref[0][2];const dy=(m[1][0]-ref[1][0])*x+(m[1][1]-ref[1][1])*y+m[1][2]-ref[1][2];arrow(x*550/1024,y*550/1024,dx*100*550/1024,dy*100*550/1024)}
fc.font='17px system-ui';fc.fillStyle='#eee';fc.fillText(selector.value,590,60);fc.fillText('Arrow magnification: 100x',590,100);const s=D.differences[selector.value];fc.fillText('RMS: '+s.rms_px.toFixed(4)+' px',590,140);fc.fillText('Max: '+s.max_px.toFixed(4)+' px',590,175);arrow(600,220,.1*100*550/1024,0);fc.fillStyle='#eee';fc.fillText('0.1 px',630,227)}
selector.onchange=mode.onchange=gain.oninput=draw;
setInterval(()=>{if(mode.value==='blink'){blink=!blink;draw()}},600);
</script></html>"""
    template = template.replace("__ROWS__", rows)
    template = template.replace("__EXAMPLE__", str(report["example_frame_id"]))
    template = template.replace("__PAYLOAD__", json.dumps(payload))
    (OUTPUT / "comparison.html").write_text(template, encoding="utf-8")


def write_markdown(report):
    original = report["parameters"]["calib_json"]
    independent = report["parameters"]["sift_similarity"]
    delta = report["parameter_deltas_vs_json"]["sift_similarity"]
    difference = report["comparison_to_json"]
    matrices = report["matrices_flipped_red_to_green"]
    rows = [
        "| 参数 | 原 JSON | 独立相似变换 | 差值（新 − 原） |",
        "|---|---:|---:|---:|",
    ]
    for key, label in [("scale", "缩放"), ("rotation_deg", "旋转（°）"),
                       ("tx", "X 平移（px）"), ("ty", "Y 平移（px）")]:
        rows.append(f"| {label} | {original[key]:.9f} | {independent[key]:.9f} | {delta[key]:+.9f} |")
    text = f"""# 红绿通道 registration matrix 比较

独立相似变换与 `calib_params.json` 的全场位置差异 RMS 为
**{difference['sift_similarity']['rms_px']:.4f} px**，最大
**{difference['sift_similarity']['max_px']:.4f} px**。
当前仓库默认 MIP / rigid ECC 与 JSON 的全场差异 RMS 为
**{difference['pipeline_ecc']['rms_px']:.4f} px**，最大
**{difference['pipeline_ecc']['max_px']:.4f} px**。

输入：`{DATA_ROOT}`，254 对 `(Y,X)=(1024,1024)` uint16 单帧 TIFF。
所有输出位于当前工作区；原始数据及 JSON 未写入。

## 统一坐标后的同模型比较

方向为 **水平翻转后的红通道 → 原始绿通道**，坐标为零起点像素中心 `(x,y)`，
水平翻转 `x'=1023-x`。角度沿用原 JSON 的 OpenCV 符号约定。

{chr(10).join(rows)}

原矩阵：

```text
{np.array2string(np.array(matrices['calib_json']), precision=9)}
```

独立相似变换：

```text
{np.array2string(np.array(matrices['sift_similarity']), precision=9)}
```

## 当前提信号代码使用的矩阵

代码中的方向为 **原始红通道 → 水平翻转后的绿通道**。
若 `F=[[-1,0,1023],[0,1,0],[0,0,1]]`，则 `E=F @ M @ F`（以齐次矩阵计算）。
两套矩阵不能逐项直接比较。

原 JSON 转成代码方向：

```text
{np.array2string(np.array(report['matrices_red_to_flipped_green']['calib_json']), precision=9)}
```

实际调用当前仓库 `estimate_sampled_registration_matrix` 的输出：

```text
{np.array2string(np.array(report['matrices_red_to_flipped_green']['pipeline_ecc']), precision=9)}
```

`registration_matrix.npy` 保存后者，可被当前提信号 API 读取。
`calib_json_red_to_flipped_green.npy` 保存原 JSON 转换到当前代码约定的矩阵。
`sift_similarity_red_to_flipped_green.npy` 保存独立相似变换转换到代码约定的矩阵。
这些文件均不会自动替换提信号配置。

## 拟合与验证

- 随机种子 {SEED}。20 帧分块依据采集设置的扫描周期；12 个完整块中选
  {NUM_FIT_BLOCKS} 块运行原项目的 full-block MIP / Euclidean ECC，10/10 通过一致性筛选。
- 原始数据缺少逐帧 Z 坐标和 z-step，20 帧块仅用于复现图像聚合算法，
  不确认其为完整三维体积。没有把 z-slice 当作神经时间点。
- 相似变换在每个拟合块中选一张原始帧，以双向 SIFT ratio={SIFT_RATIO}、
  RANSAC={RANSAC_THRESHOLD_PX} px 独立拟合，再对 10 个结果的缩放、角度、tx、ty 取中位数。
- 保留另外两块及末尾 14 帧，共 {report['validation']['num_frames']} 帧作技术验证，
  未参与任何最终矩阵的拟合。逐帧结果及原始特征对应坐标已保存 CSV。
- 图像相关在相同有效区域内计算，先作 sigma=1.5 px 高斯平滑；特征误差
  来自留出帧各自独立 RANSAC 的内点，不是人工地标真值。
- 所有帧来自同一标定板和同次记录，属于技术重复，没有独立生物学重复或 p 值。
  未知样品平面的像素尺寸，所有几何误差仅报告 px。

| 方法 | 平均逐帧特征 RMSE（px） | 平均 NCC |
|---|---:|---:|
"""
    for name, stats in report["validation"]["summary"].items():
        text += f"| {name} | {stats['mean_frame_feature_rmse_px']:.4f} | {stats['mean_ncc']:.6f} |\n"
    text += f"""
只水平翻转、不再作几何变换时，平均 NCC 为
{report['validation']['flip_only_mean_ncc']:.6f}。

默认 ECC 固定缩放为 1，原 JSON 缩放为 {original['scale']:.9f}，
因此默认 ECC 与原 JSON 的差异包含模型自由度差异及 MIP、预处理、图像相似性目标的影响；
不能仅凭 NCC 判定哪个几何变换更准确，也不能把这些差异单独归因于硬件漂移。

交互图：`comparison.html`；四面板图：`comparison_overview.png`；
完整参数、输入清单、元数据与脚本哈希：`comparison.json`、`source_files.csv`。

复现：`pixi run python scripts/compare_calibration_20261104.py`。
"""
    (OUTPUT / "report.md").write_text(text, encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(4)
    cv2.setRNGSeed(SEED)
    calibration_path = DATA_ROOT / "calib_params.json"
    initial_calibration_digest = digest(calibration_path)
    calibration = json.loads(calibration_path.read_text())
    settings = json.loads((DATA_ROOT / "sg_settings.sgpreset").read_text())
    block_size = int(settings["ch1_blueprint"]["parameters"]["period_in_frames"])
    red_folder = DATA_ROOT / "0_Camera-Red_VSC-10629"
    green_folder = DATA_ROOT / "1_Camera-Green_VSC-09321"
    red_files, green_files = sorted(red_folder.glob("*.tif")), sorted(green_folder.glob("*.tif"))
    if [p.name for p in red_files] != [p.name for p in green_files]:
        raise ValueError("The camera frame IDs do not match")
    first = tifffile.imread(red_files[0])
    if first.ndim != 2 or first.shape != (calibration["image_height"], calibration["image_width"]):
        raise ValueError("Raw frame dimensions disagree with the saved calibration")
    shape = first.shape
    width = shape[1]
    if calibration["direction"] != "flipped_ref->sig" or calibration["flip_x"] != 1 or calibration["flip_y"] != 0:
        raise ValueError("Unexpected calibration direction or flips")
    cameras = {
        "red": camera_metadata(red_folder, red_files),
        "green": camera_metadata(green_folder, green_files),
    }
    source_rows = []
    for channel, files in [("red", red_files), ("green", green_files)]:
        for path in files:
            stat = path.stat()
            source_rows.append({"channel": channel, "frame_id": int(path.stem),
                                "path": str(path), "bytes": stat.st_size,
                                "mtime_ns": stat.st_mtime_ns})
    write_csv(OUTPUT / "source_files.csv", source_rows)
    # Import just this numerical module. Package __init__ also loads PyTorch,
    # which introduces a second OpenMP runtime in this Windows environment.
    module_path = WORKSPACE / "src" / "signal_extraction" / "registration.py"
    spec = importlib.util.spec_from_file_location("registration_only", module_path)
    registration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registration)
    num_blocks = len(red_files) // block_size
    if num_blocks <= NUM_FIT_BLOCKS:
        raise ValueError("Need complete blocks reserved for independent validation")

    def block_loader(files, index, channel):
        def load():
            log(f"Loading {channel} MIP input block {index}")
            images = [tifffile.imread(p) for p in files[index * block_size:(index + 1) * block_size]]
            if any(im.shape != shape or im.dtype != first.dtype for im in images):
                raise ValueError("Inconsistent TIFF shape or dtype")
            return np.stack(images)
        return load

    log("Running the repository's sampled MIP / rigid ECC calculation")
    pipeline_matrix, pipeline_diagnostics = registration.estimate_sampled_registration_matrix(
        [block_loader(red_files, i, "red") for i in range(num_blocks)],
        [block_loader(green_files, i, "green") for i in range(num_blocks)],
        num_samples=NUM_FIT_BLOCKS, seed=SEED,
    )
    fit_blocks = pipeline_diagnostics["sampled_positions"]
    rng = np.random.default_rng(SEED)
    fit_ids = [int(block_size * block + rng.integers(block_size)) for block in fit_blocks]
    heldout_ids = [
        i for i in range(len(red_files))
        if i // block_size not in fit_blocks
    ]
    log(f"Similarity training frames: {fit_ids}; held-out frames: {len(heldout_ids)}")
    fit_matrices, fit_rows = [], []
    for index in fit_ids:
        red = tifffile.imread(red_files[index])[:, ::-1]
        green = tifffile.imread(green_files[index])
        matrix, _, _, details = feature_fit(red, green)
        fit_matrices.append(matrix)
        fit_rows.append({"frame_id": index, "block_id": index // block_size,
                         **parameters(matrix), **details})
    similarity = median_similarity(fit_matrices)
    original = np.asarray(calibration["M"], dtype=float)
    matrices = {
        "calib_json": original,
        "sift_similarity": similarity,
        "pipeline_ecc": flipped_convention(pipeline_matrix, width),
    }
    differences = {
        name: displacement_summary(m, original, shape) for name, m in matrices.items()
    }
    feature_errors = {name: [] for name in matrices}
    score_rows, point_rows, failed_feature_ids = [], [], []
    example_id = heldout_ids[len(heldout_ids) // 2]
    for position, index in enumerate(heldout_ids):
        red = tifffile.imread(red_files[index])[:, ::-1]
        green = tifffile.imread(green_files[index])
        image_scores = evaluate_images(red, green, matrices)
        row = {"frame_id": index, "block_id": index // block_size,
               "frame_phase_mod20": index % block_size,
               **{name + "_ncc": score for name, score in image_scores.items()}}
        try:
            local_matrix, source, target, details = feature_fit(red, green)
            row["num_feature_inliers"] = len(source)
            for name, matrix in matrices.items():
                error = np.linalg.norm(apply_points(matrix, source) - target, axis=1)
                feature_errors[name].append(error)
                row[name + "_feature_rmse_px"] = float(np.sqrt(np.mean(error**2)))
            for a, b in zip(source, target):
                point_rows.append({
                    "frame_id": index, "source_flipped_red_x": a[0],
                    "source_flipped_red_y": a[1], "target_green_x": b[0],
                    "target_green_y": b[1],
                })
        except (ValueError, cv2.error) as error:
            failed_feature_ids.append(index)
            row["num_feature_inliers"] = 0
            for name in matrices:
                row[name + "_feature_rmse_px"] = None
            log(f"Feature validation failed for frame {index}: {error}")
        score_rows.append(row)
        if position % 10 == 0:
            log(f"Validated {position + 1}/{len(heldout_ids)} held-out frames")
    summary = {}
    for name in matrices:
        errors = np.concatenate(feature_errors[name])
        frame_rmse = [r[name + "_feature_rmse_px"] for r in score_rows
                      if r[name + "_feature_rmse_px"] is not None]
        ncc = np.array([r[name + "_ncc"] for r in score_rows])
        summary[name] = {
            "num_matched_points": len(errors),
            "pooled_feature_rmse_px": float(np.sqrt(np.mean(errors**2))),
            "median_feature_error_px": float(np.median(errors)),
            "mean_frame_feature_rmse_px": float(np.mean(frame_rmse)),
            "median_frame_feature_rmse_px": float(np.median(frame_rmse)),
            "mean_ncc": float(ncc.mean()), "median_ncc": float(np.median(ncc)),
            "min_ncc": float(ncc.min()), "max_ncc": float(ncc.max()),
        }
    if digest(calibration_path) != initial_calibration_digest:
        raise RuntimeError("Source calibration changed during the analysis")
    if any(Path(row["path"]).stat().st_mtime_ns != row["mtime_ns"] for row in source_rows):
        raise RuntimeError("A source TIFF changed during the analysis")
    report = {
        "created_at_Asia_Shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "data_root": str(DATA_ROOT),
        "coordinate_convention": {
            "array_axes": "YX", "shape": list(shape), "dtype": str(first.dtype),
            "point_order": "zero-based pixel-center (x,y)",
            "main_matrix": "horizontally_flipped_red -> original_green",
            "pipeline_matrix": "original_red -> horizontally_flipped_green",
            "horizontal_flip": "x_flipped = image_width - 1 - x",
            "matrix_conversion": "M_pipeline = F @ M_json @ F; F=[[-1,0,W-1],[0,1,0],[0,0,1]]",
            "rotation_deg": "-atan2(M[1,0],M[0,0]) * 180/pi, matching original JSON",
            "pixel_size_um": None, "z_step_um": None, "frame_to_z": "unconfirmed",
            "block_interpretation": "20-frame nonoverlapping scan-cycle blocks, not confirmed volumes",
        },
        "camera_metadata": cameras,
        "calibration_sha256": initial_calibration_digest,
        "settings_sha256": digest(DATA_ROOT / "sg_settings.sgpreset"),
        "registration_module_sha256": digest(module_path),
        "analysis_script_sha256": digest(Path(__file__)),
        "software": {"python": sys.version, "numpy": np.__version__,
                     "opencv": cv2.__version__, "tifffile": tifffile.__version__},
        "original_calibration": calibration,
        "fit": {
            "seed": SEED, "block_size_frames": block_size,
            "num_complete_blocks": num_blocks,
            "num_partial_tail_frames": len(red_files) % block_size,
            "fit_block_ids": fit_blocks,
            "similarity_fit_frame_ids": fit_ids,
            "similarity_estimator": "median scale, angle, tx, ty over per-frame mutual SIFT / RANSAC fits",
            "sift_ratio": SIFT_RATIO, "ransac_threshold_px": RANSAC_THRESHOLD_PX,
            "per_frame_similarity": fit_rows,
        },
        "pipeline_diagnostics": pipeline_diagnostics,
        "matrices_flipped_red_to_green": {n: m.tolist() for n, m in matrices.items()},
        "matrices_red_to_flipped_green": {
            n: flipped_convention(m, width).tolist() for n, m in matrices.items()
        },
        "parameters": {n: parameters(m) for n, m in matrices.items()},
        "parameter_deltas_vs_json": {
            n: {key: parameters(m)[key] - parameters(original)[key] for key in parameters(m)}
            for n, m in matrices.items()
        },
        "comparison_to_json": differences,
        "validation": {
            "heldout_frame_ids": heldout_ids,
            "heldout_block_ids": sorted(set(range(num_blocks)) - set(fit_blocks)),
            "num_frames": len(heldout_ids),
            "feature_failure_frame_ids": failed_feature_ids,
            "correspondence_definition": "mutual SIFT ratio matches, independently filtered per validation frame by a 2px similarity RANSAC; not manual ground truth",
            "ncc_definition": "Pearson correlation after sigma=1.5px Gaussian smoothing, shared eroded valid overlap for all matrices",
            "flip_only_mean_ncc": float(np.mean([r["flip_only_ncc"] for r in score_rows])),
            "summary": summary,
            "independence": "Technical frames of one recording / one target; no biological replication or inferential p-values",
        },
        "example_frame_id": example_id,
        "source_calibration_unchanged": True,
        "source_tiff_mtimes_unchanged": True,
    }
    write_csv(OUTPUT / "similarity_training_frames.csv", fit_rows)
    write_csv(OUTPUT / "heldout_frame_metrics.csv", score_rows)
    write_csv(OUTPUT / "heldout_feature_correspondences.csv", point_rows)
    for name, matrix in matrices.items():
        np.save(OUTPUT / f"{name}_flipped_red_to_green.npy", matrix)
        np.save(OUTPUT / f"{name}_red_to_flipped_green.npy",
                flipped_convention(matrix, width))
    np.save(OUTPUT / "registration_matrix.npy", pipeline_matrix)
    independent_calibration = {
        **parameters(similarity), "flip_x": 1, "flip_y": 0,
        "M": similarity.tolist(), "image_width": width, "image_height": shape[0],
        "direction": calibration["direction"],
        "estimator": report["fit"]["similarity_estimator"],
        "num_fit_frames": len(fit_ids),
        "per_frame_inliers": [r["num_inliers"] for r in fit_rows],
    }
    (OUTPUT / "independent_calib_params.json").write_text(
        json.dumps(independent_calibration, indent=2), encoding="utf-8"
    )
    (OUTPUT / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    red_example = tifffile.imread(red_files[example_id])[:, ::-1]
    green_example = tifffile.imread(green_files[example_id])
    make_overview(red_example, green_example, matrices, original, OUTPUT / "comparison_overview.png")
    for name, matrix in matrices.items():
        cv2.imwrite(str(OUTPUT / f"overlay_{name}.png"), overlay(red_example, green_example, matrix))
    write_html(report, red_example, green_example, matrices)
    write_markdown(report)
    log(json.dumps({
        "parameters": report["parameters"], "displacement": differences,
        "heldout_summary": summary, "pipeline_matrix": pipeline_matrix.tolist(),
        "output": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
