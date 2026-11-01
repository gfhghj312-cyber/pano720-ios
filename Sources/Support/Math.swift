import Foundation
import simd

typealias Vec3 = SIMD3<Float>

// MARK: - 球面方向工具
// 世界坐标系：Z 轴垂直向上（CoreMotion .xArbitraryZVertical 参考系），
// 经度 lon 绕 Z 轴，纬度 lat 为仰角。等距圆柱（equirectangular）投影：
//   lon = (u - 0.5) * 2π ,  lat = (0.5 - v) * π

@inline(__always) func dirFrom(pitch: Float, yaw: Float) -> Vec3 {
    let cp = cos(pitch)
    return Vec3(cp * cos(yaw), cp * sin(yaw), sin(pitch))
}

@inline(__always) func pitchOf(_ v: Vec3) -> Float {
    asin(max(-1, min(1, v.z)))
}

@inline(__always) func yawOf(_ v: Vec3) -> Float {
    atan2(v.y, v.x)
}

@inline(__always) func angleBetween(_ a: Vec3, _ b: Vec3) -> Float {
    let d = simd_dot(simd_normalize(a), simd_normalize(b))
    return acos(max(-1, min(1, d)))
}

@inline(__always) func wrapAngle(_ a: Float) -> Float {
    var x = a
    while x > .pi { x -= 2 * .pi }
    while x < -.pi { x += 2 * .pi }
    return x
}

@inline(__always) func rad2deg(_ r: Float) -> Float { r * 180 / .pi }

@inline(__always) func clampf(_ v: Float, _ lo: Float, _ hi: Float) -> Float {
    min(max(v, lo), hi)
}

extension simd_float3x3 {
    /// 设备坐标系 -> 世界坐标系的旋转
    static func from(quaternion q: simd_quatf) -> simd_float3x3 {
        simd_float3x3(q)
    }

    /// 相机光轴方向（设备 -Z 方向即后置摄像头朝向）
    var cameraForward: Vec3 {
        -(self * Vec3(0, 0, 1))
    }

    /// 三行（用于传给 Metal，按 float4 对齐）
    var rows4: [SIMD4<Float>] {
        let t = transpose
        return [
            SIMD4<Float>(t.columns.0, 0),
            SIMD4<Float>(t.columns.1, 0),
            SIMD4<Float>(t.columns.2, 0)
        ]
    }
}

/// 绕 Z 轴旋转
func rotationZ(_ angle: Float) -> simd_float3x3 {
    simd_float3x3(simd_quatf(angle: angle, axis: Vec3(0, 0, 1)))
}
