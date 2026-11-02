import Foundation
import CoreGraphics
import ImageIO
import simd

/// 用 OpenCV 的特征匹配做相对姿态估计，再全局求解每帧的修正量。
///
/// 这是整套拼接的对齐引擎。之前用「直接法光度配准」，0.8° 以上的误差就掉进局部极小值，
/// 而且对曝光变化、纹理重复极度敏感 —— 这就是真实场景里重影反复出现的原因。
///
/// 这里换成成熟做法（OpenCV 特征匹配 + RANSAC 单应，也就是 Hugin/Street View 那一类的
/// 前端几何估计）：
/// 1. 每帧最多和 4 个视场最近的邻居做 ORB/AKAZE 匹配，RANSAC 解单应，
///    再由 H = K·R·K⁻¹ 反解出相对旋转（纯旋转模型下单应与深度无关，所以这一步是精确的）。
/// 2. 把「R_jᵀR_i = M_ij」当约束，反复线性化求解 3N 个修正量（旋转平均），
///    环路的累积漂移会被一并分配掉，不会像链式传播那样越走越偏。
/// 3. 用配对点的亮度比重解逐帧增益。
///
/// 安全阀：解出来的姿态如果和陀螺仪差太远（>25°），整段丢弃，退回陀螺仪姿态。
/// 这样即使特征匹配被重复纹理骗了，也不会比不用更差。
enum FeatureAligner {

    struct Result {
        var rotations: [simd_float3x3]
        var gains: [Float]
        var pairs: Int
        var residualDeg: Double
        var gyroDeltaDeg: Double
    }

    private struct Edge {
        let i: Int
        let j: Int
        let measured: simd_float3x3   // R_jᵀ · R_i（把 i 的射线转到 j）
        let ratio: Float
        let inliers: Int
    }

    // MARK: - 入口

    static func align(inputs: [StitchInput],
                      maxNeighbors: Int = 4,
                      progress: ((Double) -> Void)? = nil) -> Result? {

        let n = inputs.count
        guard n >= 3, PanoCVMatcher.isAvailable() else { return nil }

        // 1) 统一尺寸的缩略图（特征匹配不需要高分辨率）
        var images: [CGImage] = []
        images.reserveCapacity(n)
        for input in inputs {
            guard let cg = thumbnail(url: input.url, maxPixel: 600) else { return nil }
            images.append(cg)
        }
        progress?(0.02)

        // 2) 候选帧对：按陀螺仪姿态找视场相交的最近邻居，控制匹配次数量级
        var seen = Set<Int64>()
        var order: [(Int, Int)] = []
        for i in 0..<n {
            let reachI = halfDiagonal(inputs[i])
            var near: [(Int, Float)] = []
            for j in 0..<n where j != i {
                let ang = angleBetween(inputs[i].rotation.cameraForward,
                                       inputs[j].rotation.cameraForward)
                if ang < (reachI + halfDiagonal(inputs[j])) * 1.05 {
                    near.append((j, ang))
                }
            }
            near.sort { $0.1 < $1.1 }
            for (j, _) in near.prefix(maxNeighbors) {
                let a = min(i, j), b = max(i, j)
                let key = (Int64(a) << 32) | Int64(b)
                if seen.insert(key).inserted { order.append((a, b)) }
            }
        }
        let minEdges = max(3, n / 4)
        guard order.count >= minEdges else { return nil }

        // 3) 逐对匹配
        var edges: [Edge] = []
        edges.reserveCapacity(order.count)
        for (idx, pair) in order.enumerated() {
            defer {
                if idx % 4 == 0 {
                    progress?(0.02 + 0.60 * Double(idx + 1) / Double(order.count))
                }
            }
            let a = pair.0, b = pair.1
            guard let p = PanoCVMatcher.matchImageA(images[a],
                                                    imageB: images[b],
                                                    tanHalfH: Double(inputs[a].tanHalfH),
                                                    tanHalfV: Double(inputs[a].tanHalfV)) else {
                continue
            }
            edges.append(Edge(i: a, j: b,
                              measured: p.rotation,
                              ratio: Float(p.brightnessRatio),
                              inliers: Int(p.inliers)))
        }
        guard edges.count >= minEdges else { return nil }
        progress?(0.64)

        // 4) 安全阀（一）：特征相对旋转必须和陀螺仪大体一致。
        //    坐标约定写错时这里会报出 ~180°，直接放弃而不是把错误姿态喂给拼接。
        let gyroDeltas = edges.map { e -> Float in
            let gyro = inputs[e.j].rotation.transpose * inputs[e.i].rotation
            return angleOf(gyro.transpose * e.measured) * 180 / .pi
        }
        let gyroMedian = median(gyroDeltas)
        guard gyroMedian < 25 else { return nil }

        // 5) 旋转平均：反复线性化，解 3N 个局部修正量
        //    迭代上限给到 12：加了阻尼之后每轮只走一部分，6 轮收敛不干净。
        var R = inputs.map(\.rotation)
        for _ in 0..<12 {
            let dim = 3 * n
            var A = [Float](repeating: 0, count: dim * dim)
            var rhs = [Float](repeating: 0, count: dim)

            for e in edges {
                let i = e.i, j = e.j
                let ai = 3 * i, aj = 3 * j
                let At = e.measured.transpose          // 线性化里的 A = Mᵀ
                let C = R[j].transpose * R[i]
                let r = logRotation(At * C)
                let w: Float = 1

                for k in 0..<3 {
                    A[(ai + k) * dim + (ai + k)] += w
                    A[(aj + k) * dim + (aj + k)] += w
                    rhs[ai + k] += -w * r[k]
                }
                // 耦合块：J_i = I，J_j = -A
                for rr in 0..<3 {
                    for cc in 0..<3 {
                        let m = At[cc][rr]                 // At[rr][cc]
                        A[(ai + rr) * dim + (aj + cc)] += -w * m
                        A[(aj + cc) * dim + (ai + rr)] += -w * m
                    }
                }
                let atr = At * r
                for k in 0..<3 { rhs[aj + k] += w * atr[k] }
            }

            // 阻尼是必须的，不是保守：环状图里存在一类「整体缓慢旋转」的弱约束方向，
            // 结构矩阵在那上面接近奇异。不压住的话步长会逐轮指数放大
            // （实测 3.6°→2.3°→2.6°→3.1°→4.1°→5.4°→7.2°），第 3 轮之后整圈姿态就飞了。
            // 仿真标定：lambda≈0.6 时 cost 0.0152→0.0028，相对姿态误差 0.79°→0.43°。
            for k in 0..<dim { A[k * dim + k] += 0.6 }
            // 钉住第 0 帧（只消除整体规范自由度）
            for k in 0..<3 { A[k * dim + k] += 1e4 }

            guard let delta = solve(&A, &rhs, dim) else { return nil }

            var move: Float = 0
            let limit: Float = 5 * .pi / 180
            for i in 0..<n {
                var d = SIMD3<Float>(delta[3 * i], delta[3 * i + 1], delta[3 * i + 2])
                let len = simd_length(d)
                if len > limit { d = d / len * limit }
                move = max(move, simd_length(d))
                R[i] = R[i] * FrameAligner.rotation(from: d)
            }
            if move < 1e-4 { break }
        }
        progress?(0.78)

        // 6) 安全阀（二）：单帧修正量不能太离谱
        let drift = (0..<n).map { i -> Float in
            angleOf(inputs[i].rotation.transpose * R[i]) * 180 / .pi
        }
        guard median(drift) < 25 else { return nil }

        let residuals = edges.map { e -> Float in
            let C = R[e.j].transpose * R[e.i]
            return simd_length(logRotation(e.measured.transpose * C)) * 180 / .pi
        }

        let gains = solveGains(edges: edges, n: n)
        progress?(0.82)

        return Result(rotations: R,
                      gains: gains,
                      pairs: edges.count,
                      residualDeg: Double(median(residuals)),
                      gyroDeltaDeg: Double(gyroMedian))
    }

    // MARK: - 增益

    private static func solveGains(edges: [Edge], n: Int) -> [Float] {
        guard n > 0 else { return [] }
        var A = [Float](repeating: 0, count: n * n)
        var b = [Float](repeating: 0, count: n)

        for e in edges {
            let d = logf(max(1e-3, e.ratio))
            A[e.i * n + e.i] += 1
            A[e.j * n + e.j] += 1
            A[e.i * n + e.j] -= 1
            A[e.j * n + e.i] -= 1
            b[e.j] += d
            b[e.i] -= d
        }
        // 均值归零（几何平均 = 1），顺带定住规范自由度
        for i in 0..<n {
            for j in 0..<n { A[i * n + j] += 1.0 / Float(n) }
        }
        for i in 0..<n { A[i * n + i] += 1e-4 }

        guard let u = solve(&A, &b, n) else { return Array(repeating: 1, count: n) }
        return u.map { expf(max(logf(0.72), min(logf(1.40), $0))) }
    }

    // MARK: - 数学工具

    /// 旋转矩阵 -> 旋转向量
    static func logRotation(_ m: simd_float3x3) -> SIMD3<Float> {
        let tr = m[0][0] + m[1][1] + m[2][2]
        let c = max(-1, min(1, (tr - 1) * 0.5))
        let ang = acos(c)
        // R[r][c] == m[c][r]
        let vx = m[1][2] - m[2][1]
        let vy = m[2][0] - m[0][2]
        let vz = m[0][1] - m[1][0]
        if ang < 1e-5 { return SIMD3<Float>(vx, vy, vz) * 0.5 }
        let k = ang / (2 * sin(ang))
        return SIMD3<Float>(vx, vy, vz) * k
    }

    private static func angleOf(_ m: simd_float3x3) -> Float {
        let tr = m[0][0] + m[1][1] + m[2][2]
        return acos(max(-1, min(1, (tr - 1) * 0.5)))
    }

    private static func angleBetween(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        acos(max(-1, min(1, simd_dot(simd_normalize(a), simd_normalize(b)))))
    }

    private static func halfDiagonal(_ input: StitchInput) -> Float {
        atan(sqrt(input.tanHalfH * input.tanHalfH + input.tanHalfV * input.tanHalfV))
    }

    private static func median(_ a: [Float]) -> Float {
        guard !a.isEmpty else { return 0 }
        let s = a.sorted()
        return s[s.count / 2]
    }

    /// 高斯消元（列主元）
    private static func solve(_ A: inout [Float], _ b: inout [Float], _ dim: Int) -> [Float]? {
        guard dim > 0, A.count == dim * dim, b.count == dim else { return nil }
        for col in 0..<dim {
            var best = col
            var bestVal = abs(A[col * dim + col])
            for row in (col + 1)..<dim {
                let v = abs(A[row * dim + col])
                if v > bestVal { bestVal = v; best = row }
            }
            if bestVal < 1e-12 { return nil }
            if best != col {
                for k in 0..<dim { A.swapAt(col * dim + k, best * dim + k) }
                b.swapAt(col, best)
            }
            let piv = A[col * dim + col]
            for row in (col + 1)..<dim {
                let f = A[row * dim + col] / piv
                if f == 0 { continue }
                for k in col..<dim { A[row * dim + k] -= f * A[col * dim + k] }
                b[row] -= f * b[col]
            }
        }
        var x = [Float](repeating: 0, count: dim)
        for row in stride(from: dim - 1, through: 0, by: -1) {
            var s = b[row]
            for k in (row + 1)..<dim { s -= A[row * dim + k] * x[k] }
            x[row] = s / A[row * dim + row]
        }
        return x.allSatisfy { $0.isFinite } ? x : nil
    }

    private static func thumbnail(url: URL, maxPixel: Int) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixel
        ]
        return CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
    }
}
