import Foundation
import CoreGraphics
import simd

/// 球面上的一个拍摄节点（机位）
struct CaptureNode: Identifiable, Equatable {
    enum Kind: Int {
        case ring, zenith, nadir
    }

    let id: Int
    let pitch: Float          // 仰角，弧度
    let yaw: Float            // 方位角，弧度
    let kind: Kind
    var captured: Bool = false

    var direction: Vec3 { dirFrom(pitch: pitch, yaw: yaw) }

    /// 天底（正下方）是可跳过的，通常拍到的是脚/三脚架
    var isOptional: Bool { kind == .nadir }
}

enum PanoramaGrid {

    /// 依实际视场角，用「贪心球面覆盖」生成机位：
    /// 每帧的可拍范围是视场矩形（按 overlap 缩小），不断在「离所有已有机位最远的未覆盖点」
    /// 放新机位，直到整个球面被覆盖。
    ///
    /// 这样得到的机位数是接近最优的，而且**覆盖是构造保证的**——
    /// 只要用户拍完所有机位，拼接就不可能出空洞。
    ///
    /// - Parameters:
    ///   - tanHalfH: 竖屏水平半视场角正切（已乘裁切系数）
    ///   - tanHalfV: 竖屏垂直半视场角正切（已乘裁切系数）
    ///   - overlap: 相邻帧重叠比例，0.16 ≈ 16%（1× 约 35 张，0.5× 约 14 张）
    ///   - baseDirection: 第一张要正对的方向
    static func nodes(tanHalfH: Float,
                      tanHalfV: Float,
                      overlap: Float = 0.16,
                      baseDirection: Vec3 = Vec3(1, 0, 0),
                      maxNodes: Int = 72) -> [CaptureNode] {

        let hH = max(0.10, tanHalfH)
        let hV = max(0.10, tanHalfV)
        // 视场矩形按重叠比例收缩；收缩后的矩形就是「一帧保证覆盖」的范围
        let keep = 1 - clampf(overlap, 0, 0.55)
        let covH = hH * keep
        let covV = hV * keep

        // 球面采样点（Fibonacci 分布）
        let sampleCount = 5200
        var samples = [Vec3]()
        samples.reserveCapacity(sampleCount)
        let golden = Float.pi * (3 - sqrt(5))
        for i in 0..<sampleCount {
            let z = 1 - 2 * (Float(i) + 0.5) / Float(sampleCount)
            let r = sqrt(max(0, 1 - z * z))
            let th = golden * Float(i)
            samples.append(Vec3(r * cos(th), r * sin(th), z))
        }

        struct Axis {
            let forward: Vec3
            let right: Vec3
            let up: Vec3
        }

        func axis(_ direction: Vec3) -> Axis {
            let f = simd_normalize(direction)
            var hint = Vec3(0, 0, 1)
            if abs(simd_dot(f, hint)) > 0.97 { hint = Vec3(0, 1, 0) }
            let right = simd_normalize(simd_cross(hint, f))
            let up = simd_cross(f, right)
            return Axis(forward: f, right: right, up: up)
        }

        func covers(_ a: Axis, _ d: Vec3) -> Bool {
            let cz = simd_dot(d, a.forward)
            guard cz > 0.02 else { return false }
            let tx = simd_dot(d, a.right) / cz
            let ty = simd_dot(d, a.up) / cz

            // 正对天顶/天底时机身滚转角是不确定的（用户怎么举都有可能），
            // 这里按旋转不变的圆判定，代价是多覆盖一点，张数不变。
            if abs(a.forward.z) > 0.93 {
                let r = min(covH, covV)
                return tx * tx + ty * ty <= r * r
            }
            return abs(tx) <= covH && abs(ty) <= covV
        }

        var axes: [Axis] = [axis(baseDirection)]
        var covered = [Bool](repeating: false, count: sampleCount)
        var nearest = [Float](repeating: .pi, count: sampleCount)

        func apply(_ a: Axis) {
            for i in 0..<sampleCount {
                let d = samples[i]
                if !covered[i] && covers(a, d) { covered[i] = true }
                let ang = acos(max(-1, min(1, simd_dot(d, a.forward))))
                if ang < nearest[i] { nearest[i] = ang }
            }
        }
        apply(axes[0])

        while axes.count < maxNodes {
            // 找「离已有机位最远」的未覆盖点 —— 那里就是当前最大的洞
            var bestIndex = -1
            var bestAngle: Float = 0.02
            for i in 0..<sampleCount where !covered[i] {
                if nearest[i] > bestAngle {
                    bestAngle = nearest[i]
                    bestIndex = i
                }
            }
            guard bestIndex >= 0 else { break }
            let newAxis = axis(samples[bestIndex])
            axes.append(newAxis)
            apply(newAxis)
        }

        // 归类 + 排序
        var raw: [CaptureNode] = axes.map { a in
            let pitch = pitchOf(a.forward)
            let yaw = yawOf(a.forward)
            let kind: CaptureNode.Kind
            if pitch > 1.30 { kind = .zenith }
            else if pitch < -1.30 { kind = .nadir }
            else { kind = .ring }
            return CaptureNode(id: 0, pitch: pitch, yaw: yaw, kind: kind)
        }

        // 从最接近起始朝向的机位出发，按最近邻走一圈：用户转动的总角度最小
        let startDirection = simd_normalize(baseDirection)
        var order: [Int] = []
        var used = [Bool](repeating: false, count: raw.count)
        if let first = raw.indices.min(by: {
            angleBetween(raw[$0].direction, startDirection) < angleBetween(raw[$1].direction, startDirection)
        }) {
            order.append(first)
            used[first] = true
        }
        while order.count < raw.count {
            guard let last = order.last else { break }
            let from = raw[last].direction
            var pick = -1
            var bestAngle: Float = .greatestFiniteMagnitude
            for i in raw.indices where !used[i] {
                let a = angleBetween(raw[i].direction, from)
                if a < bestAngle {
                    bestAngle = a
                    pick = i
                }
            }
            guard pick >= 0 else { break }
            order.append(pick)
            used[pick] = true
        }

        return order.enumerated().map { newID, oldIndex in
            let n = raw[oldIndex]
            return CaptureNode(id: newID, pitch: n.pitch, yaw: n.yaw, kind: n.kind)
        }
    }
}
