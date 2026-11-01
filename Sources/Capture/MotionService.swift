import Foundation
import CoreMotion
import simd

struct MotionSample {
    let quaternion: simd_quatf      // 设备 -> 世界
    let rotation: simd_float3x3     // 设备 -> 世界
    let forward: Vec3               // 相机光轴（世界系）
    let timestamp: TimeInterval
}

/// 陀螺仪姿态跟踪：使用 .xArbitraryZVertical（Z 轴垂直，偏航任意但稳定，不受磁场干扰）
final class MotionService {
    private let manager = CMMotionManager()
    private(set) var isRunning = false
    var onSample: ((MotionSample) -> Void)?

    func start() {
        guard manager.isDeviceMotionAvailable, !isRunning else { return }
        manager.deviceMotionUpdateInterval = 1.0 / 60.0
        manager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] motion, _ in
            guard let self, let motion else { return }
            let q = motion.attitude.quaternion
            let sq = simd_quatf(ix: Float(q.x), iy: Float(q.y), iz: Float(q.z), r: Float(q.w))
            let m = simd_float3x3(sq)
            let forward = -(m * Vec3(0, 0, 1))
            self.onSample?(MotionSample(quaternion: sq,
                                        rotation: m,
                                        forward: forward,
                                        timestamp: motion.timestamp))
        }
        isRunning = true
    }

    func stop() {
        manager.stopDeviceMotionUpdates()
        isRunning = false
    }
}
