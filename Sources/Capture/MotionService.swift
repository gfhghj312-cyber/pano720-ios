import Foundation
import CoreMotion
import simd

struct MotionSample {
    let quaternion: simd_quatf      // 相机 -> 世界
    let rotation: simd_float3x3     // 相机 -> 世界
    let forward: Vec3               // 相机光轴（世界系）
    let timestamp: TimeInterval
}

/// 陀螺仪姿态跟踪：使用 .xArbitraryZVertical（Z 轴垂直，偏航任意但稳定，不受磁场干扰）。
///
/// 两件事必须做对，否则拼出来的全景一定对不上：
/// 1. **旋转方向**。CMAttitude 的 rotationMatrix 到底是「设备→参考系」还是「参考系→设备」，
///    文档和坊间代码互相矛盾。这里不猜：参考系里重力恒为 (0,0,-1)，
///    而 deviceMotion.gravity 是设备系下的重力，两边一比就知道该不该转置。
/// 2. **时间对齐**。保留最近几秒的姿态历史，按视频帧 PTS 取姿态（四元数插值），
///    补偿视频流 1~2 帧的缓冲；手在转动时这能差出 1~3°。
final class MotionService {
    private let manager = CMMotionManager()
    private(set) var isRunning = false
    var onSample: ((MotionSample) -> Void)?

    /// 判定结果：attitude.rotationMatrix 是否为「参考系 -> 设备」
    private(set) var attitudeIsWorldToDevice: Bool?

    private struct Entry {
        let timestamp: TimeInterval
        let quaternion: simd_quatf   // 相机 -> 世界
    }

    /// 只在主线程读写（回调已投递到 .main）
    private var entries: [Entry] = []
    private let maxEntries = 500

    func start() {
        guard manager.isDeviceMotionAvailable, !isRunning else { return }
        manager.deviceMotionUpdateInterval = 1.0 / 90.0
        manager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] motion, _ in
            guard let self, let motion else { return }

            let raw = motion.attitude.quaternion
            let rawQuat = simd_quatf(ix: Float(raw.x), iy: Float(raw.y), iz: Float(raw.z), r: Float(raw.w))
            let rawMatrix = simd_float3x3(rawQuat)

            // 用重力反推旋转方向
            let g = motion.gravity
            let gravityDevice = Vec3(Float(g.x), Float(g.y), Float(g.z))
            if simd_length(gravityDevice) > 0.35 {
                let down = Vec3(0, 0, -1)
                let errRaw = simd_length(rawMatrix * down - gravityDevice)          // R = 参考系->设备
                let errT = simd_length(rawMatrix.transpose * down - gravityDevice)   // R = 设备->参考系
                if abs(errRaw - errT) > 0.15 {
                    attitudeIsWorldToDevice = errRaw < errT
                }
            }

            // 统一成「相机(设备) -> 世界」
            let camToWorldQuat = (attitudeIsWorldToDevice ?? false) ? rawQuat.inverse : rawQuat
            let camToWorld = simd_float3x3(camToWorldQuat)
            let forward = -(camToWorld * Vec3(0, 0, 1))

            self.entries.append(Entry(timestamp: motion.timestamp, quaternion: camToWorldQuat))
            if self.entries.count > self.maxEntries {
                self.entries.removeFirst(self.entries.count - self.maxEntries)
            }

            self.onSample?(MotionSample(quaternion: camToWorldQuat,
                                        rotation: camToWorld,
                                        forward: forward,
                                        timestamp: motion.timestamp))
        }
        isRunning = true
    }

    func stop() {
        manager.stopDeviceMotionUpdates()
        isRunning = false
        entries.removeAll(keepingCapacity: true)
    }

    /// 取某一时刻（视频帧 PTS）的姿态，四元数插值。
    func rotation(at timestamp: TimeInterval) -> simd_float3x3? {
        guard let first = entries.first, let last = entries.last else { return nil }
        if timestamp <= first.timestamp { return simd_float3x3(first.quaternion) }
        if timestamp >= last.timestamp { return simd_float3x3(last.quaternion) }

        var lo = 0
        var hi = entries.count - 1
        while lo + 1 < hi {
            let mid = (lo + hi) / 2
            if entries[mid].timestamp <= timestamp { lo = mid } else { hi = mid }
        }
        let a = entries[lo]
        let b = entries[hi]
        let span = b.timestamp - a.timestamp
        guard span > 1e-6 else { return simd_float3x3(a.quaternion) }

        var t = Float((timestamp - a.timestamp) / span)
        t = min(max(t, 0), 1)

        // 四元数有双重覆盖：取短弧插值
        var qb = b.quaternion
        if simd_dot(a.quaternion.vector, qb.vector) < 0 {
            qb = simd_quatf(vector: -qb.vector)
        }
        return simd_float3x3(simd_slerp(a.quaternion, qb, t))
    }
}
