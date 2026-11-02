"""验证「姿态精配准」：陀螺仪给的初始姿态带漂移，用重叠区的光度误差把它修回来。

Swift 端 Sources/Stitch/FrameAlignment.swift 是这份代码的逐行移植。

核心：对每一对有重叠的帧 (i, j)，固定 j、只解 i 的一个小旋转 δ（外加焦距缩放 σ），
最小化 sum (I_i(p) - I_j(project(p)))²。δ 用前向差分雅可比 + Huber 权重。
然后把每帧从所有邻居得到的 δ 做阻尼平均，迭代若干轮（旋转平均，bundle adjustment 的简化版）。
"""
import numpy as np

TAU = 2 * np.pi


# ---------------------------------------------------------------- 基础

def bilinear(img, su, sv):
    h, w = img.shape
    x = np.clip(su, 0, 1) * (w - 1)
    y = np.clip(sv, 0, 1) * (h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def pixel_dir(su, sv, tan_h, tan_v):
    tx = (su - 0.5) * 2 * tan_h
    ty = (0.5 - sv) * 2 * tan_v
    c = np.stack([tx, ty, -np.ones_like(tx)], axis=-1)
    return c / np.linalg.norm(c, axis=-1, keepdims=True)


def rotmat(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def project(R, w, tan_h, tan_v):
    """世界方向 -> 该帧像素坐标（相机沿 -Z）"""
    c = w @ R
    z = -c[..., 2]
    ok = z > 1e-6
    safe = np.where(ok, z, 1.0)
    tx = c[..., 0] / safe
    ty = c[..., 1] / safe
    return 0.5 + 0.5 * tx / tan_h, 0.5 - 0.5 * ty / tan_v, ok


# ---------------------------------------------------------------- 单对求解

def sample_lattice(crop, n):
    m = 0.05
    lo, hi = 0.5 - 0.5 * crop + m, 0.5 + 0.5 * crop - m
    g = np.linspace(lo, hi, n)
    su, sv = np.meshgrid(g, g)
    return su.ravel(), sv.ravel()


def overlap_lattice(RA, RB, tan_h, tan_v, crop, n):
    """只在 A 与 B 真正重叠的那块区域取样。
    整幅撒点的话，重叠只有 15% 时 256 个点里只剩三十来个落在 B 里，
    估计出来的角度噪声就有 0.3~0.5°——精配准的精度全被这个限死了。"""
    m = 0.004
    lo_u = lo_v = 0.5 - 0.5 * crop + m
    hi_u = hi_v = 0.5 + 0.5 * crop - m
    us, vs = [], []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            tx = sx * tan_h * crop
            ty = sy * tan_v * crop
            c = np.array([tx, ty, -1.0])
            c /= np.linalg.norm(c)
            su, sv, ok = project(RA, (RB @ c)[None, :], tan_h, tan_v)
            if not ok[0]:
                return sample_lattice(crop, n)
            us.append(float(su[0]))
            vs.append(float(sv[0]))
    lo_u, hi_u = max(lo_u, min(us)), min(hi_u, max(us))
    lo_v, hi_v = max(lo_v, min(vs)), min(hi_v, max(vs))
    if hi_u - lo_u < 0.02 or hi_v - lo_v < 0.02:
        return sample_lattice(crop, n)
    gu = np.linspace(lo_u, hi_u, n)
    gv = np.linspace(lo_v, hi_v, n)
    su, sv = np.meshgrid(gu, gv)
    return su.ravel(), sv.ravel()


def overlap_fraction(imgA, RA, imgB, RB, tan_h, tan_v, crop):
    su, sv = sample_lattice(crop, 9)
    w = pixel_dir(su, sv, tan_h, tan_v) @ RA.T
    sb_u, sb_v, ok = project(RB, w, tan_h, tan_v)
    inside = ok & (sb_u > 0) & (sb_u < 1) & (sb_v > 0) & (sb_v < 1)
    return float(inside.mean())


def blur5(img):
    """可分离二项式核 [1,4,6,4,1]/16，边界钳制"""
    k = np.array([1., 4., 6., 4., 1.]) / 16

    def conv1(a, axis):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (2, 2)
        p = np.pad(a, pad, mode="edge")
        out = np.zeros_like(a)
        for t, wgt in enumerate(k):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(t, t + a.shape[axis])
            out += wgt * p[tuple(sl)]
        return out

    return conv1(conv1(img, 0), 1)


def build_pyramid(img, levels=3):
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(blur5(pyr[-1])[::2, ::2])
    return pyr


def pair_solve(imgA, RA, imgB, RB, tan_h, tan_v, crop,
               iters_fine=10, iters_coarse=14, n=22, huber=0.05, use_scale=True,
               pyrA=None, pyrB=None):
    """固定 B，解 A 的小旋转 δ 与焦距缩放 σ。返回 (δ, σ, 内点数, H=JᵀWJ, b=-JᵀWr)。
    由粗到精：粗层只管把 1~3° 的大错位拉回收敛域，细节层再修到亚像素。"""
    pyrA = pyrA or build_pyramid(imgA)
    pyrB = pyrB or build_pyramid(imgB)
    nlev = min(len(pyrA), len(pyrB))

    su, sv = overlap_lattice(RA, RB, tan_h, tan_v, crop, n)
    ia0 = bilinear(pyrA[0], su, sv)

    # 纹理门限只在最细层算一次，所有层共用同一批采样点
    gx = bilinear(pyrA[0], np.clip(su + 1.5 / pyrA[0].shape[1], 0, 1), sv) - \
         bilinear(pyrA[0], np.clip(su - 1.5 / pyrA[0].shape[1], 0, 1), sv)
    gy = bilinear(pyrA[0], su, np.clip(sv + 1.5 / pyrA[0].shape[0], 0, 1)) - \
         bilinear(pyrA[0], su, np.clip(sv - 1.5 / pyrA[0].shape[0], 0, 1))
    active = np.hypot(gx, gy) > 0.010
    if active.sum() < 24:
        return np.zeros(3), 0.0, 0, np.zeros((3, 3)), np.zeros(3)
    su, sv, ia0 = su[active], sv[active], ia0[active]
    dirs = pixel_dir(su, sv, tan_h, tan_v)

    delta = np.zeros(3)
    sigma = 0.0
    h_r = 8e-4
    h_s = 3e-3
    nparam = 4 if use_scale else 3
    Hp, bp = np.zeros((3, 3)), np.zeros(3)
    inliers = 0

    for lev in range(nlev - 1, -1, -1):
        ia = bilinear(pyrA[lev], su, sv)
        imgB_l = pyrB[lev]
        iters = iters_coarse if lev > 0 else iters_fine
        # 粗层容差放大：模糊后的残差量级本来就小
        hub = huber * (2.0 ** lev)

        def residuals(d, s):
            R2 = rotmat(d) @ RA
            w = dirs @ R2.T
            sb_u, sb_v, ok = project(RB, w, tan_h * (1 + s), tan_v * (1 + s))
            inside = ok & (sb_u > 0.003) & (sb_u < 0.997) & (sb_v > 0.003) & (sb_v < 0.997)
            r = ia - bilinear(imgB_l, np.clip(sb_u, 0, 1), np.clip(sb_v, 0, 1))
            r[~ok] = 0.0
            return r, inside

        for _ in range(iters):
            r0, inside = residuals(delta, sigma)
            inliers = int(inside.sum())
            m = inside & (np.abs(r0) < 0.30)
            if m.sum() < 24:
                break
            rr = r0[m]
            J = np.zeros((int(m.sum()), nparam))
            for k in range(3):
                dd = delta.copy()
                dd[k] += h_r
                rk, _ = residuals(dd, sigma)
                J[:, k] = (rk[m] - rr) / h_r
            if use_scale:
                rk, _ = residuals(delta, sigma + h_s)
                J[:, 3] = (rk[m] - rr) / h_s

            a = np.abs(rr)
            wgt = np.where(a <= hub, 1.0, hub / np.maximum(a, 1e-6))
            JtW = J.T * wgt
            Hf = JtW @ J + np.eye(nparam) * 1e-4
            bf = -JtW @ rr
            if lev == 0:
                Hp = Hf[:3, :3]
                bp = bf[:3]
            try:
                step = np.linalg.solve(Hf, bf)
            except np.linalg.LinAlgError:
                break
            step = np.clip(step, -0.05, 0.05)
            delta = delta + step[:3]
            if use_scale:
                sigma = float(np.clip(sigma + step[3], -0.08, 0.08))
            if np.linalg.norm(step[:3]) < 1e-6 and abs(step[3]) < 1e-5:
                break

    return delta, sigma, inliers, Hp, bp


# ---------------------------------------------------------------- 全局

def refine(imgs, R_list, tan_h, tan_v, crop, rounds=5, damp=0.85, min_overlap=0.08,
           verbose=True):
    """全局法方程：把所有成对约束装配成 3N×3N 正规方程一次解，
    相当于对全部姿态做一次 bundle adjustment（只解旋转）。"""
    n = len(imgs)
    R = [r.copy() for r in R_list]
    sigmas = []
    scales = [1.0, 1.0]
    pyr = [build_pyramid(im) for im in imgs]

    for rd in range(rounds):
        th, tv = tan_h * scales[0], tan_v * scales[1]
        cand = []
        for i in range(n):
            for j in range(i + 1, n):
                ov = overlap_fraction(imgs[i], R[i], imgs[j], R[j], th, tv, crop)
                if ov >= min_overlap:
                    cand.append((i, j, ov))

        solved = []
        for (i, j, ov) in cand:
            d, s, inl, Hp, bp = pair_solve(imgs[i], R[i], imgs[j], R[j], th, tv, crop,
                                           pyrA=pyr[i], pyrB=pyr[j])
            if inl < 30:
                continue
            solved.append((i, j, ov, d))
            if abs(s) > 1e-5:
                sigmas.append(s)

        if len(solved) < n // 2:
            break

        # 剔除离群对：重叠区被运动物体占据时会给出离谱的解
        mags = np.array([np.linalg.norm(x[3]) for x in solved])
        med = float(np.median(mags))
        limit = max(np.radians(3.0), med * 4 + np.radians(0.5))
        used = [x for x in solved if np.linalg.norm(x[3]) <= limit]

        # 「δi - δj = d」的最小二乘：3N×3N 拉普拉斯型方程，第 0 帧钉住去规范自由度
        A = np.zeros((3 * n, 3 * n))
        b = np.zeros(3 * n)
        for (i, j, ov, d) in used:
            w = ov
            ai, aj = 3 * i, 3 * j
            A[ai:ai + 3, ai:ai + 3] += np.eye(3) * w
            A[aj:aj + 3, aj:aj + 3] += np.eye(3) * w
            A[ai:ai + 3, aj:aj + 3] -= np.eye(3) * w
            A[aj:aj + 3, ai:ai + 3] -= np.eye(3) * w
            b[ai:ai + 3] += w * d
            b[aj:aj + 3] -= w * d
        A[0:3, 0:3] += np.eye(3) * 10.0

        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break

        move = 0.0
        for i in range(n):
            d = delta[3 * i:3 * i + 3] * damp
            move = max(move, float(np.linalg.norm(d)))
            R[i] = rotmat(d) @ R[i]

        # 焦距缩放：第二轮用中位数统一修正一次（这是标定偏差，不是逐帧参数）
        if rd == 1 and sigmas:
            sm = float(np.median(sigmas))
            if abs(sm) > 0.001:
                scales = [1 + sm, 1 + sm]
                if verbose:
                    print(f"    焦距修正 {sm*100:+.2f}%")

        if verbose:
            print(f"    轮 {rd+1}: {len(used)}/{len(solved)} 对约束, 中位残差 {med*180/np.pi:.3f}°, 最大修正 {np.degrees(move):.3f}°")
        if move < 2e-4:
            break

    return R, (float(np.median(sigmas)) if sigmas else 0.0), scales


def gain_solve(imgs, R_list, tan_h, tan_v, crop, iters=80):
    """用重叠区亮度比解每帧增益（比全局均值回归准）"""
    n = len(imgs)
    logs = np.zeros(n)
    obs = []
    for i in range(n):
        for j in range(i + 1, n):
            su, sv = sample_lattice(crop, 12)
            w = pixel_dir(su, sv, tan_h, tan_v) @ R_list[i].T
            sb_u, sb_v, ok = project(R_list[j], w, tan_h, tan_v)
            inside = ok & (sb_u > 0.05) & (sb_u < 0.95) & (sb_v > 0.05) & (sb_v < 0.95)
            if inside.sum() < 40:
                continue
            a = bilinear(imgs[i], su[inside], sv[inside]).mean()
            b = bilinear(imgs[j], sb_u[inside], sb_v[inside]).mean()
            if a < 0.02 or b < 0.02:
                continue
            obs.append((i, j, float(np.log(b / a))))

    for _ in range(iters):
        for (i, j, b) in obs:
            target = logs[j] + b
            logs[i] += 0.25 * (target - logs[i])
    logs -= logs.mean()
    return np.exp(np.clip(logs, np.log(0.7), np.log(1.45)))


# ---------------------------------------------------------------- 测试场景

def make_test_scene(w=4096, h=2048, seed=5):
    """带直边的场景（模拟门、窗、家具、地脚线），比纯噪声更接近真实拍摄"""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 0.62)
    for _ in range(300):
        y0 = int(rng.integers(int(0.10 * h), int(0.90 * h)))
        x0 = int(rng.integers(0, w))
        ww = int(rng.integers(24, 260))
        hh = int(rng.integers(18, 200))
        c = rng.uniform(0.10, 0.98, 3)
        xs = (np.arange(x0, x0 + ww)) % w
        img[y0:y0 + hh, xs] = c
    # 细长条纹：更容易看出亚像素错位
    for _ in range(60):
        y = int(rng.integers(int(0.15 * h), int(0.85 * h)))
        img[y:y + int(rng.integers(2, 6)), :] = rng.uniform(0.1, 0.95, 3)
    img += rng.normal(0, 0.008, img.shape)
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------- 主流程

def project_frame(img, R, tan_h, tan_v, crop, W=640, H=320):
    u = (np.arange(W) + .5) / W
    vv = (np.arange(H) + .5) / H
    uu, gv = np.meshgrid(u, vv)
    lon = (uu - .5) * TAU
    lat = (.5 - gv) * np.pi
    cl = np.cos(lat)
    wdir = np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], -1)
    c = wdir @ R
    z = -c[..., 2]
    ok = z > 1e-4
    safe = np.where(ok, z, 1.0)
    tx = c[..., 0] / safe
    ty = c[..., 1] / safe
    inside = ok & (np.abs(tx) <= tan_h * crop * 0.92) & (np.abs(ty) <= tan_v * crop * 0.92)
    su = 0.5 + 0.5 * tx / tan_h
    sv = 0.5 - 0.5 * ty / tan_v
    lum = img if img.ndim == 2 else img @ np.array([0.299, 0.587, 0.114])
    return bilinear(lum, su, sv), inside


def ghost_score(frames, Rl, tan_h, tan_v, crop):
    """重叠区一致度：同一处被两帧看到时的 RMS 差异 = 肉眼看到的「重影」强度。
    这是规范无关的，正好对应用户抱怨的现象。"""
    n = len(frames)
    P = [project_frame(frames[i], Rl[i], tan_h, tan_v, crop) for i in range(n)]
    tot = val = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            m = P[i][1] & P[j][1]
            if m.sum() < 2000:
                continue
            d = (P[i][0] - P[j][0])[m]
            w = float(m.sum())
            val += float(np.sqrt((d ** 2).mean())) * w
            tot += w
    return val / max(tot, 1) * 255


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("v", "Tools/verify_stitch.py")
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)

    np.random.seed(11)
    scene = make_test_scene()

    for label, (tan_h, tan_v, crop) in {
        "1x wide": (0.487, 0.650, 1.0),
        "0.5x ultra": (1.0, 1.333, 0.92),
    }.items():
        print(f"\n=== {label} ===")
        dirs, _, _ = v.grid_nodes(tan_h * crop, tan_v * crop, overlap=0.16)
        n = len(dirs)
        R_true = [v.camera_matrix(d) for d in dirs]

        R_meas = []
        drift = np.radians(1.8)
        for k, R in enumerate(R_true):
            err = np.array([np.radians(np.random.normal(0, 0.25)),
                            np.radians(np.random.normal(0, 0.25)),
                            drift * (k / max(1, n - 1)) + np.radians(np.random.normal(0, 0.18))])
            R_meas.append(rotmat(err) @ R)

        imgs = [v.shoot(scene, R, tan_h, tan_v, frame_w=240, frame_h=320) @ np.array([0.299, 0.587, 0.114])
                for R in R_true]
        frames = [v.shoot(scene, R, tan_h, tan_v) for R in R_true]

        def pose_err(Rl):
            e = [np.degrees(np.arccos(np.clip((np.trace(Rl[i].T @ R_true[i]) - 1) / 2, -1, 1))) for i in range(n)]
            return float(np.mean(e)), float(np.max(e))

        g_true = ghost_score(frames, R_true, tan_h, tan_v, crop)
        g_meas = ghost_score(frames, R_meas, tan_h, tan_v, crop)
        p0 = pose_err(R_meas)

        R_fix, sigma, scales = refine(imgs, R_meas, tan_h, tan_v, crop, verbose=True)
        g_fix = ghost_score(frames, R_fix, tan_h, tan_v, crop)
        p1 = pose_err(R_fix)

        print(f"  姿态误差   前 {p0[0]:.3f}/{p0[1]:.3f}   后 {p1[0]:.3f}/{p1[1]:.3f}   (焦距 {sigma*100:+.2f}%)")
        print(f"  重影 RMS   理想 {g_true:.2f}   前 {g_meas:.2f}   后 {g_fix:.2f}"
              f"   -> 减少 {(1 - g_fix / g_meas) * 100:.0f}%")


if __name__ == "__main__":
    main()
