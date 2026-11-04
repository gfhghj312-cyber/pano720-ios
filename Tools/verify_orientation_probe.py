"""验证「用重叠区一致度自动探测图像朝向」是否可靠。

背景：连续多轮在 Windows 上推断 iOS 的 CGImage 行序 / CVPixelBuffer 旋转 /
videoFieldOfView 含义，改一次错一次。这里改成让设备自己测：
把 8 种可能的朝向当成候选，逐一算重叠区两帧亮度差，最小的就是正确朝向。

判据为什么可靠：整体姿态旋转是公共的，两帧在同一个球面方向采到的像素
本来就来自同一物理点，所以这个分数与全局姿态误差无关，只反映
「相对姿态 + 内参 + 图像朝向」。方向错一个 flip/rot，分数会急剧变大。
"""
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("v", "Tools/verify_stitch.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

TAU = 2 * np.pi
NAMES = ["identity", "flipY", "rot90", "rot90m", "rot180", "flipX", "rot270", "rot270m"]


def transform(frame, idx):
    """D4 群：4 个旋转 × 是否上下镜像，覆盖 flip/rot/transpose 全部情况。"""
    k = idx // 2
    mirror = (idx % 2) == 1
    out = np.rot90(frame, k, axes=(0, 1))
    if mirror:
        out = np.flip(out, axis=0)
    return np.ascontiguousarray(out)


def bilinear(img, u, vv):
    h, w = img.shape[:2]
    x = np.clip(u, 0, 1) * (w - 1)
    y = np.clip(vv, 0, 1) * (h - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0; fy = y - y0
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def ghost_score(frames, Rl, tH, tV, n=12):
    """规范无关的重叠区一致度（越小越好）。"""
    lums = [f @ np.array([.299, .587, .114]) if f.ndim == 3 else f for f in frames]
    su, sv = np.meshgrid(np.linspace(.12, .88, n), np.linspace(.12, .88, n))
    su, sv = su.ravel(), sv.ravel()
    tx = (su - .5) * 2 * tH
    ty = (.5 - sv) * 2 * tV
    d = np.stack([tx, ty, -np.ones_like(tx)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    total, weight = 0.0, 0.0
    for i in range(len(frames)):
        ia = bilinear(lums[i], su, sv)
        for j in range(i + 1, len(frames)):
            ang = np.arccos(np.clip(Rl[i][:, 2] @ Rl[j][:, 2], -1, 1))
            if ang > (np.arctan(tH) + np.arctan(tV)) * 1.05:
                continue
            w = d @ Rl[i].T                    # 相机系 -> 世界系
            c = w @ Rl[j]                      # 世界系 -> j 的相机系
            z = -c[..., 2]
            ok = z > 1e-4
            safe = np.where(ok, z, 1.0)
            u2 = .5 + .5 * (c[..., 0] / safe) / tH
            v2 = .5 - .5 * (c[..., 1] / safe) / tV
            m = ok & (u2 > .01) & (u2 < .99) & (v2 > .01) & (v2 < .99)
            if m.sum() < 40:
                continue
            diff = ia[m] - bilinear(lums[j], u2, v2)[m]
            rms = np.sqrt((diff ** 2).mean())
            total += rms * m.sum(); weight += m.sum()
    return total / max(weight, 1) * 255


def tans_for(shape, focal):
    h, w = shape[:2]
    return (w / 2) / focal, (h / 2) / focal


def stitch_all(frames, Rl, tH, tV):
    out, _ = v.stitch(frames, Rl, tH, tV, 1.0, out_w=480, out_h=240, feather=0.07)
    return out


# 与 Swift 的 AxisCalibration.matrix 完全相同：只置换/取反相机系 x/y，保持 z。
Q_SWIFT = [
    ("identity",      [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ("flipY",         [[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    ("flipX",         [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ("rot180",        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    ("rot90cw",       [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    ("rot90ccw",      [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    ("transpose",     [[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
    ("antiTranspose", [[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
]


def check_pose_absorption():
    """验证 Swift 的真实做法：不物理摆正图像，而是把 Q 吸收进姿态 R' = R·Qᵀ。

    这样 Metal / OpenCV / 光度配准三条路径拿到的 rotation 自动一致。
    """
    np.random.seed(4)
    scene = v.make_scene(2048, 1024)
    gu, gv = np.meshgrid((np.arange(480) + .5) / 480, (np.arange(240) + .5) / 240)
    gt = v.sample_equirect(scene, (gu - .5) * TAU, (.5 - gv) * np.pi)
    tH, tV = 1.0, 1.333
    focal = 240 / (2 * tH)
    dirs, _, _ = v.grid_nodes(tH, tV, overlap=0.20)
    R = np.array([v.camera_matrix(d) for d in dirs])
    frames = [v.shoot(scene, r, tH, tV, frame_w=240, frame_h=320) for r in R]

    print("\n=== Q 吸收版（与 Swift 实现一致） ===")
    print("注入朝向    探测 Q        score   次优     PSNR(选中)  PSNR(最优)  判定")
    ok_count = 0
    for inj in range(8):
        bad = [transform(f, inj) for f in frames]
        h, w = bad[0].shape[:2]
        tHi, tVi = (w / 2) / focal, (h / 2) / focal   # 按实际 JPEG 尺寸算内参
        scores, psnrs = [], []
        for _, rows in Q_SWIFT:
            q = np.array(rows, float)
            Rp = R @ q.T
            scores.append(ghost_score(bad, Rp, tHi, tVi))
            psnrs.append(v.psnr(stitch_all(bad, Rp, tHi, tVi), gt))
        k = int(np.argmin(scores))
        second = np.sort(scores)[1]
        ok = abs(psnrs[k] - max(psnrs)) < 0.8
        ok_count += ok
        print(f"  {NAMES[inj]:<10} -> {Q_SWIFT[k][0]:<13} {scores[k]:6.2f}  {second:6.2f}"
              f"   {psnrs[k]:8.2f}   {max(psnrs):8.2f}   {'OK' if ok else 'MISS'}")
    print(f"  → {ok_count}/8 选中正确朝向")


def main():
    np.random.seed(4)
    scene = v.make_scene(2048, 1024)
    gu, gv = np.meshgrid((np.arange(480) + .5) / 480, (np.arange(240) + .5) / 240)
    gt = v.sample_equirect(scene, (gu - .5) * TAU, (.5 - gv) * np.pi)

    # 0.5x 超广角：14 帧左右，跑得快；物理焦距固定不变
    tH, tV = 1.0, 1.333
    focal = 240 / (2 * tH)
    dirs, _, _ = v.grid_nodes(tH, tV, overlap=0.20)
    Rl = [v.camera_matrix(d) for d in dirs]
    frames = [v.shoot(scene, R, tH, tV, frame_w=240, frame_h=320) for R in Rl]
    print(f"场景：{len(frames)} 帧，真值朝向 PSNR {v.psnr(stitch_all(frames, Rl, tH, tV), gt):.2f} dB\n")
    print("注入朝向   探测选中    score(选中)  score(identity)   PSNR(选中)  PSNR(最优)   判定")

    for injected in range(8):
        bad = [transform(f, injected) for f in frames]
        scores = []
        for cand in range(8):
            fixed = [transform(f, cand) for f in bad]
            ch, cv = tans_for(fixed[0].shape, focal)
            scores.append(ghost_score(fixed, Rl, ch, cv))
        pick = int(np.argmin(scores))
        fixed = [transform(f, pick) for f in bad]
        ch, cv = tans_for(fixed[0].shape, focal)
        psnr = v.psnr(stitch_all(fixed, Rl, ch, cv), gt)
        best_psnr = max(
            v.psnr(stitch_all([transform(f, c) for f in bad], Rl,
                              *tans_for(transform(bad[0], c).shape, focal)), gt)
            for c in range(8))
        ok = "OK" if abs(psnr - best_psnr) < 0.8 else "MISS"
        print(f"  {NAMES[injected]:<9} -> {NAMES[pick]:<9} "
              f"{scores[pick]:8.2f}  {scores[0]:12.2f}   {psnr:9.2f}  {best_psnr:9.2f}   {ok}")


if __name__ == "__main__":
    main()
    check_pose_absorption()
