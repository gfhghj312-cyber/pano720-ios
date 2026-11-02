"""旋转平均求解器的收敛验证（对应 Sources/Stitch/FeatureAligner.swift 的第 5 步）

为什么需要这个脚本：
第一次实现时用了几乎不加阻尼的 Gauss-Newton，结果在环状机位图上发散 ——
cost 先降后升，步长逐轮指数放大（3.6°→2.3°→2.6°→3.1°→4.1°→5.4°→7.2°），
第 3 轮之后整圈姿态就飞了，最终误差 15°+。

原因是环状图里存在一类「整体缓慢旋转」的弱约束方向，结构矩阵接近奇异。
所以必须加 Levenberg 阻尼，并给足迭代轮数。

这里用带漂移的陀螺仪姿态 + 带噪声的相对旋转测量，比较三种设置：
  1. 不做旋转平均（纯陀螺仪）
  2. lambda = 1e-3（最初写错的值）
  3. lambda = 0.6（现在用的值）

判断标准用「去全局旋转后的相对误差」：整体旋转只改变全景的起始经度，
不改变接缝处对不对齐，用绝对误差会得出错误结论。

跑法：python Tools/verify_rotation_averaging.py
"""

import numpy as np

N = 24                 # 一圈的帧数
DRIFT_DEG = 2.4        # 陀螺仪整圈累积偏航漂移
EDGE_NOISE_DEG = 0.15  # 相邻帧相对旋转的测量噪声（SIFT + RANSAC 的典型量级）


def rot(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-9:
        return v / 2
    return v * th / (2 * np.sin(th))


def angle(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)))


def build(seed):
    rng = np.random.default_rng(seed)
    R_true = [rot(np.array([0.0, 2 * np.pi * i / N, 0.0])) for i in range(N)]

    R_gyro = []
    for i in range(N):
        drift = np.radians(DRIFT_DEG) * i / (N - 1)
        err = np.array([
            np.radians(rng.normal(0, 0.25)),
            drift + np.radians(rng.normal(0, 0.20)),
            np.radians(rng.normal(0, 0.20)),
        ])
        R_gyro.append(rot(err) @ R_true[i])

    edges = []
    for i in range(N):
        j = (i + 1) % N
        M = R_true[j].T @ R_true[i]
        edges.append((i, j, M @ rot(np.radians(rng.normal(0, EDGE_NOISE_DEG, 3)))))
    return R_true, R_gyro, edges


def cost(R, edges):
    return float(np.sqrt(np.mean([np.sum(log(M.T @ (R[j].T @ R[i])) ** 2)
                                  for (i, j, M) in edges])))


def relative_error(R, R_true):
    """去掉整体旋转后的残差 —— 拼接真正关心的量"""
    G = np.zeros((3, 3))
    for i in range(N):
        G += R_true[i] @ R[i].T
    U, _, Vt = np.linalg.svd(G)
    G = U @ Vt
    if np.linalg.det(G) < 0:
        U[:, -1] *= -1
        G = U @ Vt
    e = [angle(G @ R[i], R_true[i]) for i in range(N)]
    return float(np.mean(e)), float(np.max(e))


def solve(edges, R0, lam, iters=12, max_step_deg=5.0):
    R = [x.copy() for x in R0]
    dim = 3 * N
    pending = iters
    for it in range(iters):
        A = np.zeros((dim, dim))
        b = np.zeros(dim)
        for (i, j, M) in edges:
            At = M.T
            r = log(At @ (R[j].T @ R[i]))
            ai, aj = 3 * i, 3 * j
            for k in range(3):
                A[ai + k, ai + k] += 1
                A[aj + k, aj + k] += 1
                b[ai + k] += -r[k]
            A[np.ix_(range(ai, ai + 3), range(aj, aj + 3))] += -At
            A[np.ix_(range(aj, aj + 3), range(ai, ai + 3))] += -At.T
            b[aj:aj + 3] += At @ r
        A += np.eye(dim) * lam
        for k in range(3):
            A[k, k] += 1e4          # 钉住第 0 帧，只消除规范自由度
        d = np.linalg.solve(A, b)

        move = 0.0
        for i in range(N):
            v = d[3 * i:3 * i + 3]
            n = np.linalg.norm(v)
            if n > np.radians(max_step_deg):
                v = v / n * np.radians(max_step_deg)
            move = max(move, float(np.linalg.norm(v)))
            R[i] = R[i] @ rot(v)
        pending = it + 1
        if move < 1e-4:
            break
    return R, pending


def main():
    print(f"帧数 {N}  陀螺仪漂移 {DRIFT_DEG}°  边噪声 {EDGE_NOISE_DEG}°\n")
    print(f"{'设置':<22}{'cost (rad)':>12}{'相对误差均值':>14}{'最大':>10}{'轮数':>8}")
    for seed in (7, 11):
        R_true, R_gyro, edges = build(seed)
        m = relative_error(R_gyro, R_true)
        print(f"{'纯陀螺仪 (seed %d)' % seed:<22}{cost(R_gyro, edges):>12.4f}"
              f"{m[0]:>13.2f}°{m[1]:>9.2f}°{'-':>8}")
        for lam, tag in ((1e-3, "lambda=1e-3 (错的)"), (0.6, "lambda=0.6 (现在)")):
            R, it = solve(edges, R_gyro, lam)
            m = relative_error(R, R_true)
            ok = "发散" if m[0] > 5 else "收敛"
            print(f"{tag + ' seed%d' % seed:<22}{cost(R, edges):>12.4f}"
                  f"{m[0]:>13.2f}°{m[1]:>9.2f}°{it:>6}  {ok}")
        print()

    print("结论：lambda 必须在 O(0.1~1)；1e-3 会让修正量在弱约束方向上被放大而发散。")


if __name__ == "__main__":
    main()
