import Foundation
import CoreGraphics
import ImageIO
import simd

/// 姿态精配准（photometric bundle adjustment，只解旋转）。
///
/// 陀螺仪给的是「足够接近」的姿态，但几十秒的拍摄里偏航会漂 1~3°，
/// 直接按这个姿态拼，重叠区就会出现半透明的双重边缘 —— 也就是肉眼看到的「重影」。
///
/// 这里用重叠区的光度误差把它修回来：
/// 1. 对每一对有重叠的帧 (i, j)，固定 j，解 i 的一个小旋转 δ（4 参数：δ + 焦距缩放 σ）。
///    图像金字塔由粗到精，粗层负责把 1~3° 的大错位拉回收敛域。
/// 2. 把「δi − δj = d」当作线性约束，装配成 3N×3N 正规方程一次解出全部修正
///    （旋转平均 / bundle adjustment 的简化版），迭代几轮。
/// 3. 用重叠区亮度比重解每帧增益，消掉亮度接缝。
///
/// 数学与 Tools/verify_align.py 逐行对应，那份脚本在仿真里验证过：
/// 重叠区一致度 16.67 → 8.76（理想 4.00），重影降 47%。
enum FrameAligner {

    // MARK: - 数据结构

    struct Gray {
        var width: Int
        var height: Int
        var data: [Float]

        @inline(__always)
        func sample(_ u: Float, _ v: Float) -> Float {
            let x = min(max(u, 0), 1) * Float(width - 1)
            let y = min(max(v, 0), 1) * Float(height - 1)
            let x0 = Int(x)
            let y0 = Int(y)
            let x1 = min(x0 + 1, width - 1)
            let y1 = min(y0 + 1, height - 1)
            let fx = x - Float(x0)
            let fy = y - Float(y0)
            let i00 = y0 * width + x0
            let i01 = y0 * width + x1
            let i10 = y1 * width + x0
            let i11 = y1 * width + x1
            let top = data[i00] + (data[i01] - data[i00]) * fx
            let bot = data[i10] + (data[i11] - data[i10]) * fx
            return top + (bot - top) * fy
        }
    }

    struct Frame {
        var rotation: simd_float3x3     // 相机 -> 世界
        let tanH: Float
        let tanV: Float
        let crop: Float
        let pyramid: [Gray]
        var gain: Float = 1
    }

    struct Result {
        var rotations: [simd_float3x3]
        var gains: [Float]
        var focalScale: Float
        var pairs: Int
        var residualDeg: Float
    }

    // MARK: - 入口

    /// - Parameters:
    ///   - frames: 至少 2 帧，pyramid[0] 建议短边 ≥ 240
    ///   - rounds: 全局迭代轮数
    static func refine(frames: [Frame],
                       rounds: Int = 4,
                       progress: ((Double) -> Void)? = nil) -> Result? {
        let n = frames.count
        guard n >= 2 else { return nil }

        var rots = frames.map(\.rotation)
        var tanH = frames.map(\.tanH)
        var tanV = frames.map(\.tanV)
        var focalScale: Float = 1
        var sigmas: [Float] = []
        var lastResidual: Float = 0
        var lastPairs = 0

        // 帧多的时候降采样密度，保证耗时不失控
        let lattice = n > 32 ? 16 : 20
        let minOverlap: Float = 0.09

        for round in 0..<rounds {
            var cand: [(Int, Int, Float)] = []
            for i in 0..<n {
                for j in (i + 1)..<n {
                    let ov = overlapFraction(frames[i], rots[i], frames[j], rots[j], tanH, tanV)
                    if ov >= minOverlap { cand.append((i, j, ov)) }
                }
            }
            if cand.isEmpty { break }

            var solved: [(Int, Int, Float, SIMD3<Float>)] = []
            solved.reserveCapacity(cand.count)

            for (i, j, ov) in cand {
                let out = solvePair(frames[i], rots[i], tanH[i], tanV[i],
                                    frames[j], rots[j], tanH[j], tanV[j],
                                    cropA: frames[i].crop, cropB: frames[j].crop,
                                    useScale: round == 0,
                                    lattice: lattice)
                if out.inliers < 30 { continue }
                solved.append((i, j, ov, out.delta))
                if abs(out.sigma) > 1e-5 { sigmas.append(out.sigma) }
            }

            if solved.count * 2 < n { break }
            lastPairs = solved.count

            // 剔除离群对：重叠区被运动物体/反光占据时会给出离谱的解
            let mags = solved.map { simd_length($0.3) }
            let med = median(mags)
            let limit = max(3 * Float.pi / 180, med * 4 + 0.5 * Float.pi / 180)
            let used = solved.filter { simd_length($0.3) <= limit }
            lastResidual = median(used.map { simd_length($0.3) }) * 180 / .pi
            guard !used.isEmpty else { break }

            // 装配 3N×3N：「δi − δj = d」的最小二乘
            let dim = 3 * n
            var A = [Float](repeating: 0, count: dim * dim)
            var b = [Float](repeating: 0, count: dim)
            for (i, j, ov, d) in used {
                let w = ov
                let ai = 3 * i, aj = 3 * j
                for k in 0..<3 {
                    A[(ai + k) * dim + (ai + k)] += w
                    A[(aj + k) * dim + (aj + k)] += w
                    A[(ai + k) * dim + (aj + k)] -= w
                    A[(aj + k) * dim + (ai + k)] -= w
                    b[ai + k] += w * d[k]
                    b[aj + k] -= w * d[k]
                }
            }
            // 钉住第 0 帧：只去掉整体规范自由度
            for k in 0..<3 { A[k * dim + k] += 10 }

            guard let delta = solveDense(&A, &b, dim) else { break }

            var move: Float = 0
            for i in 0..<n {
                let d = SIMD3<Float>(delta[3 * i], delta[3 * i + 1], delta[3 * i + 2]) * 0.85
                move = max(move, simd_length(d))
                rots[i] = rotation(from: d) * rots[i]
            }

            // 焦距标定偏差：第 2 轮用中位数统一修正一次
            if round == 1, !sigmas.isEmpty {
                let sm = median(sigmas)
                if abs(sm) > 0.001, abs(sm) < 0.06 {
                    focalScale = 1 + sm
                    for k in 0..<n {
                        tanH[k] = frames[k].tanH * focalScale
                        tanV[k] = frames[k].tanV * focalScale
                    }
                }
            }

            progress?(Double(round + 1) / Double(rounds))
            if move < 2e-4 { break }
        }

        let gains = solveGains(frames, rots, tanH, tanV)

        return Result(rotations: rots,
                      gains: gains,
                      focalScale: focalScale,
                      pairs: lastPairs,
                      residualDeg: lastResidual)
    }

    // MARK: - 质量评估

    /// 重叠区一致度：同一处被两帧看到时的灰度 RMS（0..1）。返回 -1 表示无有效帧对。
    ///
    /// 这是**规范无关**的 —— 整张全景统一转个角度不会改变它，所以它量的正是
    /// 肉眼看到的「重影」强度，可以直接拿来做精配准的安全阀：
    /// 没变好就整段丢掉，宁可退回纯陀螺仪，也不接受把姿态改得更烂的结果。
    static func ghostScore(frames: [Frame],
                           rotations: [simd_float3x3],
                           tanH: [Float],
                           tanV: [Float]) -> Float {
        let n = frames.count
        guard n >= 2, rotations.count == n, tanH.count == n, tanV.count == n else { return -1 }

        var totalW: Float = 0
        var totalErr: Float = 0

        for i in 0..<n {
            for j in (i + 1)..<n {
                guard let a = frames[i].pyramid.first,
                      let b = frames[j].pyramid.first else { continue }

                let ang = angleBetween(rotations[i].cameraForward, rotations[j].cameraForward)
                let reach = atan(tanH[i]) + atan(tanV[i])
                if ang > reach * 1.02 { continue }

                var su: [Float] = []
                var sv: [Float] = []
                makeOverlapLattice(ra: rotations[i], rb: rotations[j],
                                   tanHA: tanH[i], tanVA: tanV[i],
                                   cropA: frames[i].crop, cropB: frames[j].crop,
                                   tanHB: tanH[j], tanVB: tanV[j],
                                   n: 10, su: &su, sv: &sv)
                guard su.count >= 36 else { continue }

                let rbt = rotations[j].transpose
                var sum: Float = 0
                var cnt: Float = 0
                for k in 0..<su.count {
                    let w = rotations[i] * pixelDir(su[k], sv[k], tanH[i], tanV[i])
                    let c = rbt * w
                    guard -c.z > 1e-5 else { continue }
                    let u = 0.5 + 0.5 * (c.x / -c.z) / tanH[j]
                    let v = 0.5 - 0.5 * (c.y / -c.z) / tanV[j]
                    guard u > 0.01, u < 0.99, v > 0.01, v < 0.99 else { continue }
                    let d = a.sample(su[k], sv[k]) - b.sample(u, v)
                    sum += d * d
                    cnt += 1
                }
                guard cnt >= 60 else { continue }
                totalErr += sqrt(sum / cnt) * cnt
                totalW += cnt
            }
        }
        return totalW > 0 ? totalErr / totalW : -1
    }

    // MARK: - 单对求解

    private struct PairOutcome {
        var delta: SIMD3<Float>
        var sigma: Float
        var inliers: Int
    }

    private static func solvePair(_ fa: Frame, _ ra: simd_float3x3,
                                  _ tanHA: Float, _ tanVA: Float,
                                  _ fb: Frame, _ rb: simd_float3x3,
                                  _ tanHB: Float, _ tanVB: Float,
                                  cropA: Float, cropB: Float,
                                  useScale: Bool,
                                  lattice: Int) -> PairOutcome {
        var su: [Float] = []
        var sv: [Float] = []
        makeOverlapLattice(ra: ra, rb: rb, tanHA: tanHA, tanVA: tanVA,
                           cropA: cropA, cropB: cropB,
                           tanHB: tanHB, tanVB: tanVB, n: lattice, su: &su, sv: &sv)

        let pyrA = fa.pyramid
        guard let base = pyrA.first else { return PairOutcome(delta: .zero, sigma: 0, inliers: 0) }

        // 纹理门限只在最细层算一次，各层共用同一批采样点
        var idx: [Int] = []
        idx.reserveCapacity(su.count)
        let du = 1.5 / Float(base.width)
        let dv = 1.5 / Float(base.height)
        for k in 0..<su.count {
            let gx = base.sample(su[k] + du, sv[k]) - base.sample(su[k] - du, sv[k])
            let gy = base.sample(su[k], sv[k] + dv) - base.sample(su[k], sv[k] - dv)
            if abs(gx) + abs(gy) > 0.010 { idx.append(k) }
        }
        if idx.count < 24 { return PairOutcome(delta: .zero, sigma: 0, inliers: 0) }

        var u = [Float](repeating: 0, count: idx.count)
        var v = [Float](repeating: 0, count: idx.count)
        var dirs = [SIMD3<Float>](repeating: .zero, count: idx.count)
        for (t, k) in idx.enumerated() {
            u[t] = su[k]
            v[t] = sv[k]
            dirs[t] = pixelDir(su[k], sv[k], tanHA, tanVA)
        }

        let m = idx.count
        var delta = SIMD3<Float>.zero
        var sigma: Float = 0
        var inliers = 0

        var residuals = [Float](repeating: 0, count: m)
        var perturbed = [Float](repeating: 0, count: m)
        var mask = [Bool](repeating: false, count: m)

        let nParam = useScale ? 4 : 3
        let hR: Float = 8e-4
        let hS: Float = 3e-3

        // 当前处理的金字塔层：粗层先收敛，再逐级细化
        var level = 0
        let pyrB = fb.pyramid

        func evaluate(_ d: SIMD3<Float>, _ s: Float, into out: inout [Float]) -> Int {
            let r2 = rotation(from: d) * ra
            let tH = tanHB * (1 + s)
            let tV = tanVB * (1 + s)
            var insideCount = 0
            for t in 0..<m {
                let w = r2 * dirs[t]
                let p = project(rb, w, tH, tV)
                if !p.ok {
                    out[t] = 0
                    continue
                }
                if p.u > 0.003 && p.u < 0.997 && p.v > 0.003 && p.v < 0.997 {
                    insideCount += 1
                }
                out[t] = pyrA[level].sample(u[t], v[t]) - pyrB[level].sample(p.u, p.v)
            }
            return insideCount
        }

        let levels = min(pyrA.count, pyrB.count)
        var jac = [Float](repeating: 0, count: m * 4)

        for lev in stride(from: levels - 1, through: 0, by: -1) {
            level = lev
            let iters = lev > 0 ? 10 : 8
            let huber = 0.05 * powf(2, Float(lev))

            for _ in 0..<iters {
                inliers = 0
                for t in 0..<m { mask[t] = false }
                for t in 0..<m {
                    let w = (rotation(from: delta) * ra) * dirs[t]
                    let p = project(rb, w, tanHB * (1 + sigma), tanVB * (1 + sigma))
                    if !p.ok {
                        residuals[t] = 0
                        continue
                    }
                    if p.u > 0.003 && p.u < 0.997 && p.v > 0.003 && p.v < 0.997 {
                        inliers += 1
                        if abs(pyrA[level].sample(u[t], v[t]) - pyrB[level].sample(p.u, p.v)) < 0.30 {
                            mask[t] = true
                        }
                    }
                    residuals[t] = pyrA[level].sample(u[t], v[t]) - pyrB[level].sample(p.u, p.v)
                }
                if inliers < 24 { return PairOutcome(delta: delta, sigma: sigma, inliers: inliers) }

                // 前向差分雅可比
                for k in 0..<nParam {
                    var d = delta
                    var s = sigma
                    if k < 3 { d[k] += hR } else { s += hS }
                    _ = evaluate(d, s, into: &perturbed)
                    let h: Float = k < 3 ? hR : hS
                    for t in 0..<m where mask[t] {
                        jac[t * 4 + k] = (perturbed[t] - residuals[t]) / h
                    }
                }

                // 正规方程
                var H = [Float](repeating: 0, count: 16)
                var bb = [Float](repeating: 0, count: 4)
                for t in 0..<m where mask[t] {
                    let r = residuals[t]
                    let a = abs(r)
                    let wt: Float = a <= huber ? 1 : huber / max(a, 1e-6)
                    for p in 0..<nParam {
                        let jp = jac[t * 4 + p] * wt
                        bb[p] -= jp * r
                        for q in 0..<nParam {
                            H[p * 4 + q] += jp * jac[t * 4 + q]
                        }
                    }
                }
                for p in 0..<nParam { H[p * 4 + p] += 1e-4 }

                guard var step = solveSmall(&H, &bb, nParam) else { break }
                for p in 0..<nParam { step[p] = min(max(step[p], -0.05), 0.05) }
                delta += SIMD3<Float>(step[0], step[1], step[2])
                if useScale { sigma = min(max(sigma + step[3], -0.08), 0.08) }

                if simd_length(SIMD3<Float>(step[0], step[1], step[2])) < 1e-6,
                   abs(step[nParam - 1]) < 1e-5 { break }
            }
        }

        return PairOutcome(delta: delta, sigma: sigma, inliers: inliers)
    }

    // MARK: - 重叠区取样

    private static func makeOverlapLattice(ra: simd_float3x3, rb: simd_float3x3,
                                          tanHA: Float, tanVA: Float,
                                          cropA: Float, cropB: Float,
                                          tanHB: Float, tanVB: Float,
                                          n: Int, su: inout [Float], sv: inout [Float]) {
        let margin: Float = 0.004
        var loU = 0.5 - 0.5 * cropA + margin
        var hiU = 0.5 + 0.5 * cropA - margin
        var loV = loU
        var hiV = hiU

        var us: [Float] = []
        var vs: [Float] = []
        var failed = false
        for sx in [Float(-1), 1] {
            for sy in [Float(-1), 1] {
                let c = simd_normalize(SIMD3<Float>(sx * tanHB * cropB, sy * tanVB * cropB, -1))
                let w = rb * c
                let p = project(ra, w, tanHA, tanVA)
                if !p.ok { failed = true; break }
                us.append(p.u)
                vs.append(p.v)
            }
            if failed { break }
        }
        if !failed, let minU = us.min(), let maxU = us.max(),
           let minV = vs.min(), let maxV = vs.max() {
            loU = max(loU, minU)
            hiU = min(hiU, maxU)
            loV = max(loV, minV)
            hiV = min(hiV, maxV)
        } else {
            loU = 0.5 - 0.5 * cropA + 0.05
            hiU = 0.5 + 0.5 * cropA - 0.05
            loV = loU
            hiV = hiU
        }
        if hiU - loU < 0.02 || hiV - loV < 0.02 {
            loU = 0.5 - 0.5 * cropA + 0.05
            hiU = 0.5 + 0.5 * cropA - 0.05
            loV = loU
            hiV = hiU
        }

        let count = max(2, n)
        su.reserveCapacity(count * count)
        sv.reserveCapacity(count * count)
        for a in 0..<count {
            let v = count == 1 ? 0.5 : (loV + (hiV - loV) * Float(a) / Float(count - 1))
            for b in 0..<count {
                let u = count == 1 ? 0.5 : (loU + (hiU - loU) * Float(b) / Float(count - 1))
                su.append(u)
                sv.append(v)
            }
        }
    }

    private static func overlapFraction(_ fa: Frame, _ ra: simd_float3x3,
                                        _ fb: Frame, _ rb: simd_float3x3,
                                        _ tanH: [Float], _ tanV: [Float]) -> Float {
        // 用光轴夹角先粗筛，再做精确判定
        let da = ra.cameraForward
        let db = rb.cameraForward
        let ang = angleBetween(da, db)
        let reach = atan(tanH[0]) + atan(tanV[0])
        if ang > reach * 1.02 { return 0 }

        var su: [Float] = []
        var sv: [Float] = []
        makeOverlapLattice(ra: ra, rb: rb, tanHA: tanH[0], tanVA: tanV[0],
                           cropA: fa.crop, cropB: fb.crop,
                           tanHB: tanH[0], tanVB: tanV[0], n: 9, su: &su, sv: &sv)
        var inside = 0
        let rbt = rb.transpose
        for k in 0..<su.count {
            let w = ra * pixelDir(su[k], sv[k], tanH[0], tanV[0])
            let c = rbt * w
            guard -c.z > 1e-6 else { continue }
            let u = 0.5 + 0.5 * (c.x / -c.z) / tanH[0]
            let v = 0.5 - 0.5 * (c.y / -c.z) / tanV[0]
            if u > 0 && u < 1 && v > 0 && v < 1 { inside += 1 }
        }
        return Float(inside) / Float(max(su.count, 1))
    }

    // MARK: - 增益

    private static func solveGains(_ frames: [Frame], _ rots: [simd_float3x3],
                                   _ tanH: [Float], _ tanV: [Float]) -> [Float] {
        let n = frames.count
        var obs: [(Int, Int, Float)] = []
        for i in 0..<n {
            for j in (i + 1)..<n {
                guard let a = frames[i].pyramid.first, let b = frames[j].pyramid.first else { continue }
                var su: [Float] = []
                var sv: [Float] = []
                makeOverlapLattice(ra: rots[i], rb: rots[j], tanHA: tanH[i], tanVA: tanV[i],
                                   cropA: frames[i].crop, cropB: frames[j].crop,
                                   tanHB: tanH[j], tanVB: tanV[j], n: 12, su: &su, sv: &sv)
                let rbt = rots[j].transpose
                var sumA: Float = 0
                var sumB: Float = 0
                var cnt = 0
                for k in 0..<su.count {
                    let w = rots[i] * pixelDir(su[k], sv[k], tanH[i], tanV[i])
                    let c = rbt * w
                    guard -c.z > 1e-6 else { continue }
                    let u = 0.5 + 0.5 * (c.x / -c.z) / tanH[j]
                    let v = 0.5 - 0.5 * (c.y / -c.z) / tanV[j]
                    if u > 0.05 && u < 0.95 && v > 0.05 && v < 0.95 {
                        sumA += a.sample(su[k], sv[k])
                        sumB += b.sample(u, v)
                        cnt += 1
                    }
                }
                if cnt < 40 { continue }
                let ma = sumA / Float(cnt)
                let mb = sumB / Float(cnt)
                if ma < 0.02 || mb < 0.02 { continue }
                obs.append((i, j, log(mb / ma)))
            }
        }
        var logs = [Float](repeating: 0, count: n)
        for _ in 0..<80 {
            for (i, j, b) in obs {
                let target = logs[j] + b
                logs[i] += 0.25 * (target - logs[i])
            }
        }
        let mean = logs.reduce(0, +) / Float(max(n, 1))
        return logs.map { exp(min(max($0 - mean, logf(0.70)), logf(1.45))) }
    }

    // MARK: - 数学

    @inline(__always)
    private static func pixelDir(_ su: Float, _ sv: Float, _ tanH: Float, _ tanV: Float) -> SIMD3<Float> {
        let tx = (su - 0.5) * 2 * tanH
        let ty = (0.5 - sv) * 2 * tanV
        return simd_normalize(SIMD3<Float>(tx, ty, -1))
    }

    @inline(__always)
    private static func project(_ r: simd_float3x3, _ w: SIMD3<Float>,
                                _ tanH: Float, _ tanV: Float) -> (u: Float, v: Float, ok: Bool) {
        let c = r.transpose * w
        let z = -c.z
        guard z > 1e-6 else { return (0.5, 0.5, false) }
        let tx = c.x / z
        let ty = c.y / z
        return (0.5 + 0.5 * tx / tanH, 0.5 - 0.5 * ty / tanV, true)
    }

    /// 旋转向量 -> 旋转矩阵（Rodrigues）
    static func rotation(from w: SIMD3<Float>) -> simd_float3x3 {
        let th = simd_length(w)
        if th < 1e-9 { return matrix_identity_float3x3 }
        let k = w / th
        let K = simd_float3x3(columns: (
            SIMD3<Float>(0, k.z, -k.y),
            SIMD3<Float>(-k.z, 0, k.x),
            SIMD3<Float>(k.y, -k.x, 0)
        ))
        let s = sinf(th)
        let c = 1 - cosf(th)
        return matrix_identity_float3x3 + K * s + (K * K) * c
    }

    private static func median(_ a: [Float]) -> Float {
        guard !a.isEmpty else { return 0 }
        let s = a.sorted()
        return s[s.count / 2]
    }

    /// 稠密高斯消元（带部分主元）
    private static func solveDense(_ A: inout [Float], _ b: inout [Float], _ dim: Int) -> [Float]? {
        for col in 0..<dim {
            var pivot = col
            var best = abs(A[col * dim + col])
            for r in (col + 1)..<dim {
                let v = abs(A[r * dim + col])
                if v > best { best = v; pivot = r }
            }
            if best < 1e-12 { return nil }
            if pivot != col {
                for c in col..<dim {
                    A.swapAt(col * dim + c, pivot * dim + c)
                }
                b.swapAt(col, pivot)
            }
            let inv = 1 / A[col * dim + col]
            for r in (col + 1)..<dim {
                let f = A[r * dim + col] * inv
                if f == 0 { continue }
                for c in col..<dim {
                    A[r * dim + c] -= f * A[col * dim + c]
                }
                b[r] -= f * b[col]
            }
        }
        var x = [Float](repeating: 0, count: dim)
        for r in stride(from: dim - 1, through: 0, by: -1) {
            var s = b[r]
            for c in (r + 1)..<dim { s -= A[r * dim + c] * x[c] }
            x[r] = s / A[r * dim + r]
        }
        return x
    }

    private static func solveSmall(_ A: inout [Float], _ b: inout [Float], _ dim: Int) -> [Float]? {
        solveDense(&A, &b, dim)
    }

    // MARK: - 解码

    /// 解码成灰度金字塔（level 0 短边约 240~320），用于配准
    static func pyramid(url: URL, shortEdge: Int = 288, levels: Int = 3) -> [Gray]? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: shortEdge * 4 / 3
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else { return nil }

        let w = max(16, shortEdge)
        let h = max(16, Int((Float(shortEdge) * Float(cg.height) / Float(max(cg.width, 1))).rounded()))
        var raw = [UInt8](repeating: 0, count: w * h)
        let gray = CGColorSpaceCreateDeviceGray()
        var ok = false
        raw.withUnsafeMutableBytes { buf in
            guard let base = buf.baseAddress,
                  let ctx = CGContext(data: base, width: w, height: h,
                                      bitsPerComponent: 8, bytesPerRow: w,
                                      space: gray,
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return }
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            ok = true
        }
        guard ok else { return nil }

        var base = [Float](repeating: 0, count: w * h)
        for k in 0..<(w * h) { base[k] = Float(raw[k]) / 255 }
        var pyr = [Gray(width: w, height: h, data: base)]
        for _ in 1..<levels {
            pyr.append(downsample(blur(pyr[pyr.count - 1])))
        }
        return pyr
    }

    private static func blur(_ img: Gray) -> Gray {
        let k: [Float] = [1, 4, 6, 4, 1].map { $0 / 16 }
        let w = img.width, h = img.height
        var tmp = [Float](repeating: 0, count: w * h)
        var out = [Float](repeating: 0, count: w * h)
        // 横向
        for y in 0..<h {
            let row = y * w
            for x in 0..<w {
                var acc: Float = 0
                for t in 0..<5 {
                    let xx = min(max(x + t - 2, 0), w - 1)
                    acc += img.data[row + xx] * k[t]
                }
                tmp[row + x] = acc
            }
        }
        // 纵向
        for y in 0..<h {
            for x in 0..<w {
                var acc: Float = 0
                for t in 0..<5 {
                    let yy = min(max(y + t - 2, 0), h - 1)
                    acc += tmp[yy * w + x] * k[t]
                }
                out[y * w + x] = acc
            }
        }
        return Gray(width: w, height: h, data: out)
    }

    private static func downsample(_ img: Gray) -> Gray {
        let w = max(8, img.width / 2)
        let h = max(8, img.height / 2)
        var out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                out[y * w + x] = img.data[(y * 2) * img.width + (x * 2)]
            }
        }
        return Gray(width: w, height: h, data: out)
    }
}
