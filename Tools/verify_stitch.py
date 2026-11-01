"""闭环验证：合成全景 -> 按机位反投影出帧（模拟拍摄）-> 用与 Metal 着色器相同的公式拼回。

用来确认三件事：
1. 投影/逆投影的旋转方向（R 还是 Rᵀ）
2. 机位网格是否真的不留空洞
3. 羽化 + cos³ 混合的实际接缝误差
"""
import sys
import numpy as np

TAU = 2 * np.pi

# ---------------------------------------------------------------- 场景

def make_scene(w=2048, h=1024, seed=7):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:h, 0:w]

    # 低频色块 + 高频噪点：既能看接缝错位，也能量 PSNR
    lat = (0.5 - ys / h) * np.pi
    lon = (xs / w - 0.5) * TAU
    base = np.stack([
        0.5 + 0.45 * np.cos(4 * lon) * np.cos(2 * lat),
        0.5 + 0.45 * np.sin(3 * lon + 1.0) * np.cos(3 * lat),
        0.5 + 0.45 * np.cos(5 * lat + 2.0) * np.sin(2 * lon),
    ], axis=-1)

    grid = ((xs // 32 + ys // 32) % 2)[..., None] * 0.18
    noise = rng.random((h, w, 1)) * 0.16
    scene = np.clip(base + grid + noise - 0.12, 0, 1)
    return scene


def sample_equirect(scene, lon, lat):
    h, w = scene.shape[:2]
    u = lon / TAU + 0.5
    v = 0.5 - lat / np.pi
    u = u % 1.0
    v = np.clip(v, 0, 1)

    x = u * (w - 1)
    y = v * (h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = (x0 + 1) % w
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    x0 %= w

    c00 = scene[y0, x0]
    c01 = scene[y0, x1]
    c10 = scene[y1, x0]
    c11 = scene[y1, x1]
    top = c00 * (1 - fx) + c01 * fx
    bot = c10 * (1 - fx) + c11 * fx
    return top * (1 - fy) + bot * fy


# ---------------------------------------------------------------- 机位网格

def grid_nodes(tan_h, tan_v, overlap=0.24, base=(1.0, 0.0, 0.0), max_nodes=72):
    keep = 1 - overlap
    cov_h = tan_h * keep
    cov_v = tan_v * keep

    n = 5200
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(np.maximum(0, 1 - z * z))
    th = np.pi * (3 - np.sqrt(5)) * np.arange(n)
    samples = np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)

    def frame(f):
        f = f / np.linalg.norm(f)
        hint = np.array([0, 0, 1.0])
        if abs(np.dot(f, hint)) > 0.97:
            hint = np.array([0, 1.0, 0])
        right = np.cross(hint, f)
        right /= np.linalg.norm(right)
        up = np.cross(f, right)
        return f, right, up

    def covers(f, right, up, d):
        cz = d @ f
        ok = cz > 0.02
        tx = (d @ right) / np.where(ok, cz, 1.0)
        ty = (d @ up) / np.where(ok, cz, 1.0)
        return ok & (np.abs(tx) <= cov_h) & (np.abs(ty) <= cov_v)

    axes = [frame(np.array(base, dtype=float))]
    covered = covers(axes[0][0], axes[0][1], axes[0][2], samples)
    nearest = np.arccos(np.clip(samples @ axes[0][0], -1, 1))

    while len(axes) < max_nodes:
        cand = np.where(~covered)[0]
        if cand.size == 0:
            break
        idx = cand[np.argmax(nearest[cand])]
        if nearest[idx] <= 0.02:
            break
        f, right, up = frame(samples[idx])
        axes.append((f, right, up))
        covered |= covers(f, right, up, samples)
        nearest = np.minimum(nearest, np.arccos(np.clip(samples @ f, -1, 1)))

    return [a[0] for a in axes], samples, covered


def camera_matrix(f):
    f = f / np.linalg.norm(f)
    hint = np.array([0, 0, 1.0])
    if abs(np.dot(f, hint)) > 0.97:
        hint = np.array([0, 1.0, 0])
    right = np.cross(hint, f)
    right /= np.linalg.norm(right)
    up = np.cross(f, right)
    # 列 = 相机轴在世界系中的方向；-Z 看向 f
    return np.stack([right, up, -f], axis=1)


# ---------------------------------------------------------------- 拍摄

def shoot(scene, R, tan_h, tan_v, frame_w=480, frame_h=640):
    u = (np.arange(frame_w) + 0.5) / frame_w
    v = (np.arange(frame_h) + 0.5) / frame_h
    uu, vv = np.meshgrid(u, v)
    tx = (2 * uu - 1) * tan_h
    ty = (1 - 2 * vv) * tan_v
    c = np.stack([tx, ty, -np.ones_like(tx)], axis=-1)
    w = c @ R.T
    w /= np.linalg.norm(w, axis=-1, keepdims=True)
    lon = np.arctan2(w[..., 1], w[..., 0])
    lat = np.arcsin(np.clip(w[..., 2], -1, 1))
    return sample_equirect(scene, lon, lat)


# ---------------------------------------------------------------- 拼接（镜像着色器）

def stitch(frames, R_list, tan_h, tan_v, crop, out_w=1024, out_h=512,
           transpose_bug=False, relaxed=True, feather=0.11):
    u = (np.arange(out_w) + 0.5) / out_w
    v = (np.arange(out_h) + 0.5) / out_h
    uu, vv = np.meshgrid(u, v)
    lon = (uu - 0.5) * TAU
    lat = (0.5 - vv) * np.pi
    cl = np.cos(lat)
    wdir = np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1)
    wdir /= np.linalg.norm(wdir, axis=-1, keepdims=True)

    acc = np.zeros((out_h, out_w, 3))
    wsum = np.zeros((out_h, out_w))

    cover_h = tan_h * crop
    cover_v = tan_v * crop

    def sample_frame(img, su, sv):
        fh, fw = img.shape[:2]
        x = np.clip(su, 0, 1) * (fw - 1)
        y = np.clip(sv, 0, 1) * (fh - 1)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.minimum(x0 + 1, fw - 1)
        y1 = np.minimum(y0 + 1, fh - 1)
        fx = (x - x0)[..., None]
        fy = (y - y0)[..., None]
        top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
        bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
        return top * (1 - fy) + bot * fy

    for img, R in zip(frames, R_list):
        M = R if transpose_bug else R.T
        c = wdir @ M.T
        cz = c[..., 2]
        vis = cz < -1e-4
        safe = np.where(vis, -cz, 1.0)
        tx = c[..., 0] / safe
        ty = c[..., 1] / safe

        inside = vis & (np.abs(tx) <= cover_h) & (np.abs(ty) <= cover_v)
        su = 0.5 + 0.5 * tx / tan_h
        sv = 0.5 - 0.5 * ty / tan_v
        eu = 0.5 * crop - np.abs(su - 0.5)
        ev = 0.5 * crop - np.abs(sv - 0.5)
        feather_w = np.clip(eu / feather, 0, 1) * np.clip(ev / feather, 0, 1)
        feather_w = feather_w * feather_w * (3 - 2 * feather_w)

        cos2 = 1.0 / (1 + tx * tx + ty * ty)
        wt = feather_w * cos2 * np.sqrt(cos2)
        m = inside & (wt > 1e-5)

        if m.any():
            col = sample_frame(img, np.where(m, su, 0.5), np.where(m, sv, 0.5))
            acc += col * (wt * m)[..., None]
            wsum += wt * m

    holes = wsum <= 1e-5
    if relaxed and holes.any():
        for img, R in zip(frames, R_list):
            M = R if transpose_bug else R.T
            c = wdir @ M.T
            cz = c[..., 2]
            vis = cz < -1e-4
            safe = np.where(vis, -cz, 1.0)
            tx = c[..., 0] / safe
            ty = c[..., 1] / safe
            over = np.maximum(np.abs(tx) / cover_h, np.abs(ty) / cover_v)
            k = np.clip(1 - (np.maximum(over, 1.0) - 1) / 0.45, 0, 1) ** 2
            m = holes & vis & (over <= 1.45) & (k > 1e-4)
            if m.any():
                su = 0.5 + 0.5 * tx / tan_h
                sv = 0.5 - 0.5 * ty / tan_v
                cos2 = 1.0 / (1 + tx * tx + ty * ty)
                wt = k * cos2 * np.sqrt(cos2) * 0.35
                col = sample_frame(img, np.where(m, su, 0.5), np.where(m, sv, 0.5))
                acc += col * (wt * m)[..., None]
                wsum += wt * m
        holes = wsum <= 1e-5

    out = np.where(holes[..., None], 0.0, acc / np.maximum(wsum, 1e-5)[..., None])
    return out, holes.mean()


def psnr(a, b, mask=None):
    d = (a - b) ** 2
    if mask is not None:
        d = d[mask]
    mse = d.mean()
    return 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)


# ---------------------------------------------------------------- 主流程

def run(lens, tan_h, tan_v, crop, label):
    scene = make_scene()
    dirs, samples, covered = grid_nodes(tan_h * crop, tan_v * crop)
    print(f"\n=== {label} ===")
    print(f"  视场: 水平 {np.degrees(2*np.arctan(tan_h*crop)):.1f}°  "
          f"垂直 {np.degrees(2*np.arctan(tan_v*crop)):.1f}°   机位数: {len(dirs)}")

    # 起始朝向对齐经度 0（等价于 alignToFirst）
    R_list = [camera_matrix(d) for d in dirs]
    yaw0 = np.arctan2(R_list[0][:, 2][1] * -1, -R_list[0][:, 2][0])
    A = np.array([[np.cos(-yaw0), -np.sin(-yaw0), 0],
                  [np.sin(-yaw0), np.cos(-yaw0), 0],
                  [0, 0, 1]])
    R_list = [A @ R for R in R_list]

    frames = [shoot(scene, R, tan_h, tan_v) for R in R_list]

    gt_u = (np.arange(1024) + 0.5) / 1024
    gt_v = (np.arange(512) + 0.5) / 512
    gu, gv = np.meshgrid(gt_u, gt_v)
    glon = (gu - 0.5) * TAU
    glat = (0.5 - gv) * np.pi
    gt = sample_equirect(scene, glon, glat)

    out_ok, holes_ok = stitch(frames, R_list, tan_h, tan_v, crop)
    out_bad, holes_bad = stitch(frames, R_list, tan_h, tan_v, crop, transpose_bug=True)

    print(f"  空洞率(正确 Rᵀ): {holes_ok*100:.3f}%   PSNR: {psnr(out_ok, gt):.2f} dB")
    print(f"  空洞率(错误 R )：{holes_bad*100:.3f}%   PSNR: {psnr(out_bad, gt):.2f} dB")

    # 覆盖检查：球面采样点是否都被某个机位的视场矩形盖住
    print(f"  球面覆盖: {covered.mean()*100:.2f}% 个采样点落在某机位视场内")
    return out_ok, gt, len(dirs)


if __name__ == "__main__":
    run("wide", 0.487, 0.650, 1.0, "1× 广角")
    run("ultra", 1.000, 1.333, 0.88, "0.5× 超广角")
