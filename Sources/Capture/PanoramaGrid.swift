import Foundation
import CoreGraphics

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
    /// 依据实际相机视场角生成覆盖整个球面的机位网格。
    /// - Parameter overlap: 相邻帧之间的重叠比例（0.28 ≈ 28%）
    static func nodes(tanHalfH: Float, tanHalfV: Float,
                      overlap: Float = 0.28, baseYaw: Float = 0) -> [CaptureNode] {
        let hfov = 2 * atan(max(0.05, tanHalfH))
        let vfov = 2 * atan(max(0.05, tanHalfV))
        let stepH = hfov * (1 - overlap)
        let stepV = vfov * (1 - overlap)

        // 从赤道向上下依次排环
        var pitches: [Float] = [0]
        var p = stepV
        while p < Float.pi / 2 - vfov * 0.22 {
            pitches.append(p)
            pitches.append(-p)
            p += stepV
        }

        var result: [CaptureNode] = []
        var idx = 0
        for pitch in pitches {
            let cosP = max(0.12, cos(pitch))
            // 在该纬度上，保持同样的弧长间距所需的经度步长
            let stepYaw = 2 * asin(min(1, sin(stepH / 2) / cosP))
            let count = max(4, Int((2 * Float.pi / max(0.0001, stepYaw)).rounded()))
            for k in 0..<count {
                let yaw = baseYaw + Float(k) * (2 * Float.pi / Float(count))
                result.append(CaptureNode(id: idx, pitch: pitch, yaw: wrapAngle(yaw), kind: .ring))
                idx += 1
            }
        }

        result.append(CaptureNode(id: idx, pitch: Float.pi / 2 - 0.055, yaw: baseYaw, kind: .zenith))
        idx += 1
        result.append(CaptureNode(id: idx, pitch: -Float.pi / 2 + 0.055, yaw: baseYaw, kind: .nadir))
        return result
    }
}
