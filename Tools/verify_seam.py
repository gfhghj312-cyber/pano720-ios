"""验证「最优接缝（winner-take-all）」vs「所有重叠帧加权混合」。

来源：Google Street View 的 liblightcycle.so 里 rendering 层是
  seam_finder_graphcut / seam_selection / image_masker（逐像素选一帧）
  + blender 多频段金字塔（只抹平曝光差），根本没有把所有帧混合。
我们原先是加权羽化混合，两帧的同一根边缘各画一遍 -> 半透明双线重影。

本脚本用与 MSL stitchPano 逐字对应的公式对比两种模式。

## 结论（2026-11-05）：三种方案都不采用，保留原来的两频段加权混合

- 单层对比：winner 的竖线是单根锐利线，混合版发灰、变宽、有双线晕。
  但 PSNR winner 反而低 1.3~1.6 dB（多帧平均本来就有降噪效果）。
- 套上 `composeBands`（wideLow + narrow - narrowLow）后，winner 在接缝跳变处
  出现**明显白色亮斑/断裂** —— narrow 与 narrowLow 的取值在不同分辨率下
  偶发不一致，两频段相减把尖峰放大了。
- 折中方案「权重锐化」（wt^p, p=2/4/8/16/32）扫描结果单调变差：
  PSNR 17.58→17.06、P99 误差 174→180、高频能量 0.0474→0.0541（真值 0.0448）。

原因：Google 的 seam_finder 是 **Graph Cut 最小代价接缝**（接缝代价含
LaplacianCbCrDiff / ExposureUnaryCost，见 so 里的 seam_finder_utils.cc），
接缝被选在图像差异最小处；单纯「离画幅中心最近」的 Voronoi 硬切不具备这个性质。
要移植这条链路必须实现 max-flow + 接缝代价，不是改一行 shader 能解决的。
在补齐之前，保留已被验证过的窄羽化两频段混合更稳妥。
"""
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("v", "Tools/verify_stitch.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

TAU = 2 * np.pi


def stitch(frames, R_list, tan_h, tan_v, mode="blend",
           out_w=640, out_h=320, feather=0.055):
    u = (np.arange(out_w) + .5) / out_w
    vv = (np.arange(out_h) + .5) / out_h
    uu, gv = np.meshgrid(u, vv)
    lon = (uu - .5) * TAU
    lat = (.5 - gv) * np.pi
    cl = np.cos(lat)
    w = np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], -1)

    acc = np.zeros((out_h, out_w, 3))
    wsum = np.zeros((out_h, out_w))
    best = np.zeros((out_h, out_w, 3))
    bestw = np.zeros((out_h, out_w))

    for img, R in zip(frames, R_list):
        c = w @ R                              # Rᵀ·w
        z = -c[..., 2]
        ok = z > 1e-4
        safe = np.where(ok, z, 1.0)
        tx = c[..., 0] / safe
        ty = c[..., 1] / safe
        su = .5 + .5 * tx / tan_h
        sv = .5 - .5 * ty / tan_v
        eu = .5 - np.abs(su - .5)
        ev = .5 - np.abs(sv - .5)
        inside = ok & (eu > 0) & (ev > 0) & (su > 0) & (su < 1) & (sv > 0) & (sv < 1)
        f = np.clip(eu / feather, 0, 1) ** 2 * np.clip(ev / feather, 0, 1) ** 2
        cos2 = 1 / (1 + tx * tx + ty * ty)
        wt = f * cos2 ** 1.5 * inside
        col = bilinear_img(img, su, sv)
        if mode == "winner":
            m = wt > bestw
            best[m] = col[m]
            bestw[m] = wt[m]
        else:
            acc += col * wt[..., None]
            wsum += wt

    if mode == "winner":
        return best, bestw > 1e-5
    out = np.where(wsum[..., None] > 1e-5, acc / np.maximum(wsum[..., None], 1e-5), 0.07)
    return out, wsum > 1e-5


def bilinear_img(img, u, vv):
    h, w = img.shape[:2]
    x = np.clip(u, 0, 1) * (w - 1)
    y = np.clip(vv, 0, 1) * (h - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[..., None]; fy = (y - y0)[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def grad_energy(a):
    """高频能量：混合会把错位的两帧平均掉，梯度会变钝。"""
    g = a @ np.array([.299, .587, .114])
    return float(np.mean(np.abs(np.diff(g, axis=1))) + np.mean(np.abs(np.diff(g, axis=0))))


def main():
    np.random.seed(11)
    scene = v.make_scene(4096, 2048)
    OW, OH = 640, 320
    gu, gv = np.meshgrid((np.arange(OW) + .5) / OW, (np.arange(OH) + .5) / OH)
    gt = v.sample_equirect(scene, (gu - .5) * TAU, (.5 - gv) * np.pi)

    th, tv = 1.0, 1.333
    dirs, _, _ = v.grid_nodes(th, tv, overlap=0.16)
    R_true = [v.camera_matrix(d) for d in dirs]
    frames = [v.shoot(scene, R, th, tv, frame_w=240, frame_h=320) for R in R_true]

    def rot(x):
        t = np.linalg.norm(x)
        if t < 1e-12: return np.eye(3)
        k = x / t
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)

    print(f"场景 {len(frames)} 帧   真值高频能量 {grad_energy(gt):.4f}\n")
    print("姿态抖动   混合 PSNR / 高频      winner PSNR / 高频      PSNR 提升")
    for sigma in (0.0, 0.3, 0.7, 1.2):
        Rl = []
        rng = np.random.default_rng(7)
        for R in R_true:
            ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
            Rl.append(rot(ax * np.radians(sigma)) @ R)
        a, ma = stitch(frames, Rl, th, tv, "blend", OW, OH)
        b, mb = stitch(frames, Rl, th, tv, "winner", OW, OH)
        m = ma & mb
        pa = v.psnr(a, gt, m)
        pb = v.psnr(b, gt, m)
        print(f"  {sigma:4.1f}°     {pa:6.2f} / {grad_energy(a):.4f}      "
              f"{pb:6.2f} / {grad_energy(b):.4f}       {pb-pa:+5.2f} dB")


if __name__ == "__main__":
    main()
