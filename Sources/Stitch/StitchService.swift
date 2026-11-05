import Foundation
import UIKit
import Metal
import CoreGraphics
import ImageIO
import simd

struct StitchInput {
    let url: URL
    let rotation: simd_float3x3   // 相机 -> 世界
    let tanHalfH: Float           // 全画幅 竖屏水平半视场角正切
    let tanHalfV: Float           // 全画幅 竖屏垂直半视场角正切
    let crop: Float               // 参与拼接的中央比例
    let luma: Float
}

struct StitchedPanorama {
    let equirect: CGImage
    let jpeg: Data
    let thumbnail: Data
    let coverage: Double
    let seconds: TimeInterval
    let width: Int
    let height: Int
    /// 精配准后的残余姿态残差（度），越小说明重叠区对得越准
    let residualDeg: Double
    let pairCount: Int
    /// 实际生效的对齐引擎（用于结果页诊断）
    let aligner: String
    /// 真机朝向自校准结果（用于结果页诊断）
    let calibration: String
}

enum StitchError: LocalizedError {
    case noDevice
    case noInput
    case shader(String)
    case texture(String)
    case encode

    var errorDescription: String? {
        switch self {
        case .noDevice: return "此设备不支持 Metal"
        case .noInput: return "没有可拼接的画面"
        case .shader(let m): return "着色器编译失败：\(m)"
        case .texture(let m): return "纹理创建失败：\(m)"
        case .encode: return "图像编码失败"
        }
    }
}

/// 基于姿态（陀螺仪）+ 光度精配准的球面投影拼接。
///
/// 1. `FrameAligner` 用重叠区光度误差修正每帧姿态（陀螺仪漂移会让接缝出现双重边缘）。
/// 2. 对等距圆柱目标图的每个像素求世界射线 w，用每帧旋转算出相机系 c = Rᵀ·w，
///    做针孔反投影采样，权重 = 边缘羽化 × cos³(离轴角) × 逐帧增益。
/// 3. 第二轮对没覆盖到的像素做「放宽边界 + 钳制采样」的补洞。
enum StitchService {

    // MARK: - Metal 着色器（运行时编译，免去 .metal 构建配置）

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Header {
        int outW;
        int outH;
        int count;
        int pad;
    };

    struct Img {
        float4 r0;     // Rᵀ 的第 0 行（世界 -> 相机）
        float4 r1;     // Rᵀ 的第 1 行
        float4 r2;     // Rᵀ 的第 2 行
        float4 misc;   // x: tanHalfH, y: tanHalfV, z: gain, w: slice
        float4 ext;    // x: 覆盖用 tanH, y: 覆盖用 tanV, z: 0.5*crop, w: feather 宽度
    };

    // 相机看 -Z：可见点在相机系里 z < 0
    static inline float3 toCamera(Img im, float3 w) {
        return float3(dot(im.r0.xyz, w), dot(im.r1.xyz, w), dot(im.r2.xyz, w));
    }

    kernel void stitchPano(texture2d_array<float, access::sample> src [[texture(0)]],
                           texture2d<float, access::write> dst [[texture(1)]],
                           constant Header &hp [[buffer(0)]],
                           constant Img *imgs [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= uint(hp.outW) || gid.y >= uint(hp.outH)) { return; }

        constexpr sampler smp(coord::normalized, address::clamp_to_edge, filter::linear);

        float u = (float(gid.x) + 0.5) / float(hp.outW);
        float v = (float(gid.y) + 0.5) / float(hp.outH);
        float lon = (u - 0.5) * 6.28318530718;
        float lat = (0.5 - v) * 3.14159265359;
        float cl = cos(lat);
        float3 w = float3(cl * cos(lon), cl * sin(lon), sin(lat));

        float3 acc = float3(0.0);
        float wsum = 0.0;
        // 第一轮：严格落在画幅内，按到裁切边长度的距离羽化
        for (int i = 0; i < hp.count; ++i) {
            Img im = imgs[i];
            float3 c = toCamera(im, w);
            if (c.z >= -0.0001) { continue; }
            float inv = 1.0 / (-c.z);
            float tx = c.x * inv;
            float ty = c.y * inv;
            if (fabs(tx) > im.ext.x || fabs(ty) > im.ext.y) { continue; }

            float su = 0.5 + 0.5 * tx / im.misc.x;
            float sv = 0.5 - 0.5 * ty / im.misc.y;
            if (su < 0.0 || su > 1.0 || sv < 0.0 || sv > 1.0) { continue; }

            float eu = im.ext.z - fabs(su - 0.5);
            float ev = im.ext.z - fabs(sv - 0.5);
            if (eu <= 0.0 || ev <= 0.0) { continue; }

            float feather = smoothstep(0.0, im.ext.w, eu) * smoothstep(0.0, im.ext.w, ev);
            if (feather <= 0.00001) { continue; }

            // 平方一下：过渡带更集中，接缝更"窄"，不会把两帧的轻微错位摊成大片半透明重影
            float cosA2 = 1.0 / (1.0 + tx * tx + ty * ty);
            float wt = feather * feather * cosA2 * sqrt(cosA2) * im.misc.z;

            float4 col = src.sample(smp, float2(su, sv), uint(im.misc.w + 0.5));
            acc += col.rgb * wt;
            wsum += wt;
        }

        // 第二轮：仍有洞就放宽边界（最多溢到画幅外 45%），做边缘外推补洞
        if (wsum <= 0.00001) {
            for (int i = 0; i < hp.count; ++i) {
                Img im = imgs[i];
                float3 c = toCamera(im, w);
                if (c.z >= -0.0001) { continue; }
                float inv = 1.0 / (-c.z);
                float tx = c.x * inv;
                float ty = c.y * inv;

                float over = max(fabs(tx) / im.ext.x, fabs(ty) / im.ext.y);
                if (over > 1.45) { continue; }
                float k = 1.0 - (max(over, 1.0) - 1.0) / 0.45;
                k = k * k;
                if (k <= 0.0001) { continue; }

                float su = clamp(0.5 + 0.5 * tx / im.misc.x, 0.0, 1.0);
                float sv = clamp(0.5 - 0.5 * ty / im.misc.y, 0.0, 1.0);

                float cosA2 = 1.0 / (1.0 + tx * tx + ty * ty);
                float wt = k * cosA2 * sqrt(cosA2) * im.misc.z * 0.35;

                float4 col = src.sample(smp, float2(su, sv), uint(im.misc.w + 0.5));
                acc += col.rgb * wt;
                wsum += wt;
            }
        }

        float3 outc;
        float filled;
        if (wsum > 0.00001) {
            outc = acc / wsum;
            filled = 1.0;
        } else {
            outc = float3(0.07, 0.08, 0.10);
            filled = 0.0;
        }
        dst.write(float4(outc, filled), gid);
    }

    // 两频段融合：低频用「宽羽化」的合成结果（过渡平滑、看不见接缝），
    // 高频用「窄羽化」的结果减去它自己的低频（保留锐度、把重影限制在几像素内）。
    // 单频段羽化只能在「接缝明显」和「重影变宽」之间二选一，这也是之前反复翻车的原因之一。
    kernel void composeBands(texture2d<float, access::sample> narrow [[texture(0)]],
                             texture2d<float, access::sample> wideLow [[texture(1)]],
                             texture2d<float, access::sample> narrowLow [[texture(2)]],
                             texture2d<float, access::write> dst [[texture(3)]],
                             uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) { return; }
        constexpr sampler smp(coord::normalized, address::clamp_to_edge, filter::linear);

        float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
        float4 n  = narrow.sample(smp, uv);
        float3 w  = wideLow.sample(smp, uv).rgb;
        float3 nl = narrowLow.sample(smp, uv).rgb;

        float3 c = w + (n.rgb - nl);
        // alpha 保留空洞标记，后面统计覆盖率还要用
        dst.write(float4(clamp(c, 0.0, 1.0), n.a), gid);
    }
    """

    private struct Header {
        var outW: Int32 = 0
        var outH: Int32 = 0
        var count: Int32 = 0
        var pad: Int32 = 0
    }

    private struct ShaderImage {
        var r0 = SIMD4<Float>(0, 0, 0, 0)
        var r1 = SIMD4<Float>(0, 0, 0, 0)
        var r2 = SIMD4<Float>(0, 0, 0, 0)
        var misc = SIMD4<Float>(0, 0, 0, 0)
        var ext = SIMD4<Float>(0, 0, 0, 0)
    }

    /// 高频层的羽化宽度（占画幅比例）：越窄，重叠区的错位越不会被摊成大片半透明。
    private static let narrowFeather: Float = 0.055
    /// 低频层的羽化宽度：很宽，保证亮度/色温差异平滑过渡，接缝看不见。
    private static let wideFeather: Float = 0.30

    // MARK: - 入口

    static func stitch(inputs rawInputs: [StitchInput],
                       outWidth: Int = 4096,
                       progress: @escaping (Double) -> Void) async throws -> StitchedPanorama {

        guard rawInputs.count >= 1 else { throw StitchError.noInput }
        guard let device = MTLCreateSystemDefaultDevice() else { throw StitchError.noDevice }

        // 着色器只能在 async 上下文编译；后面的重活全部丢到后台线程，
        // 否则拼一张 4096×2048 会把主线程卡住，进度条也不动。
        let library: MTLLibrary
        do {
            library = try await device.makeLibrary(source: shaderSource, options: nil)
        } catch {
            throw StitchError.shader(error.localizedDescription)
        }
        guard let stitchFn = library.makeFunction(name: "stitchPano"),
              let composeFn = library.makeFunction(name: "composeBands") else {
            throw StitchError.shader("找不到 stitchPano / composeBands")
        }
        let pipes: Pipelines
        do {
            pipes = Pipelines(stitch: try await device.makeComputePipelineState(function: stitchFn),
                              compose: try await device.makeComputePipelineState(function: composeFn))
        } catch {
            throw StitchError.shader(error.localizedDescription)
        }

        return try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let out = try stitchSync(inputs: rawInputs,
                                             outWidth: outWidth,
                                             device: device,
                                             pipes: pipes,
                                             progress: progress)
                    cont.resume(returning: out)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    private struct Pipelines {
        let stitch: MTLComputePipelineState
        let compose: MTLComputePipelineState
    }

    // MARK: - 主流程（后台线程）

    private static func stitchSync(inputs rawInputs: [StitchInput],
                                   outWidth: Int,
                                   device: MTLDevice,
                                   pipes: Pipelines,
                                   progress: @escaping (Double) -> Void) throws -> StitchedPanorama {

        let start = Date()

        // 1. 让第一张的位置落在等距圆柱图的中心（经度 0）
        let firstAligned = alignToFirst(rawInputs)
        guard !firstAligned.isEmpty else { throw StitchError.noInput }

        // 1b. 在真机上实测图像朝向与焦距，不再靠桌面端推断 Core Graphics 行序 /
        //     videoRotationAngle 是否生效 / videoFieldOfView 的含义。
        //     探测结果直接吸收进每帧姿态（R' = R·Qᵀ），后面的特征匹配、光度配准、
        //     Metal 采样因此自动共用同一个正确方向。
        progress(0.01)
        let calibration = OrientationProbe.calibrate(firstAligned)
        let aligned = calibration?.inputs ?? firstAligned
        let calibrationText = calibration?.info.display ?? "朝向校准：帧数不足，未启用"

        // 2. 姿态对齐：OpenCV 特征优先 → 光度配准 → 纯陀螺仪
        progress(0.04)
        var rotations = aligned.map(\.rotation)
        var gains = aligned.map { _ in Float(1) }
        var tans = aligned.map { (h: $0.tanHalfH, v: $0.tanHalfV) }
        var residualDeg: Double = 0
        var pairCount = 0
        var aligner = "陀螺仪"

        if aligned.count >= 3 {
            if let feature = FeatureAligner.align(inputs: aligned, progress: { p in
                progress(0.04 + 0.36 * p)
            }) {
                rotations = feature.rotations
                gains = feature.gains
                residualDeg = feature.residualDeg
                pairCount = feature.pairs
                aligner = "OpenCV 特征"
            } else if let photo = photometricAlign(aligned, progress: { p in
                progress(0.42 + 0.05 * p)
            }) {
                rotations = photo.rotations
                gains = photo.gains
                residualDeg = photo.residualDeg
                pairCount = photo.pairs
                if photo.residualDeg >= 0 { aligner = "光度配准" }
                if abs(photo.focalScale - 1) > 0.001 {
                    for k in 0..<tans.count {
                        tans[k] = (h: aligned[k].tanHalfH * photo.focalScale,
                                   v: aligned[k].tanHalfV * photo.focalScale)
                    }
                }
            }
        }

        progress(0.48)

        // 3. 输出尺寸（严格 2:1）
        let outW = max(1024, outWidth - (outWidth % 2))
        let outH = outW / 2

        // 4. 源图降采样尺寸：控制显存占用
        let count = aligned.count
        let budgetBytes = 170.0 * 1024 * 1024
        let approxBytesPerTexel = 4.0 / 0.75
        var maxDim = Int(sqrt(budgetBytes / (approxBytesPerTexel * Double(count))))
        maxDim = max(900, min(1600, maxDim))

        let firstSize = imagePixelSize(of: aligned[0].url)
        let aspect = firstSize.width > 0 ? firstSize.height / firstSize.width : 4.0 / 3.0
        let texW = maxDim
        let texH = max(2, Int((Double(maxDim) * Double(aspect)).rounded()))

        // 5. 纹理数组
        let desc = MTLTextureDescriptor()
        desc.textureType = .type2DArray
        desc.pixelFormat = .rgba8Unorm
        desc.width = texW
        desc.height = texH
        desc.arrayLength = count
        desc.mipmapLevelCount = 1
        desc.usage = .shaderRead
        desc.storageMode = .shared
        guard let srcTex = device.makeTexture(descriptor: desc) else {
            throw StitchError.texture("源纹理分配失败（\(texW)x\(texH)x\(count)）")
        }

        // 6. 参数表
        var params = [ShaderImage]()
        params.reserveCapacity(count)
        var slice = 0

        for (i, input) in aligned.enumerated() {
            guard let bytes = downsampledRGBA(url: input.url, width: texW, height: texH) else { continue }

            bytes.withUnsafeBytes { raw in
                if let base = raw.baseAddress {
                    srcTex.replace(region: MTLRegionMake2D(0, 0, texW, texH),
                                   mipmapLevel: 0,
                                   slice: slice,
                                   withBytes: base,
                                   bytesPerRow: texW * 4,
                                   bytesPerImage: texW * 4 * texH)
                }
            }

            let crop = clampf(input.crop, 0.6, 1.0)
            let tanFullH = max(0.05, tans[i].h)
            let tanFullV = max(0.05, tans[i].v)

            var p = ShaderImage()
            let rows = rotations[i].worldToCameraRows4
            p.r0 = rows[0]
            p.r1 = rows[1]
            p.r2 = rows[2]
            p.misc = SIMD4<Float>(tanFullH, tanFullV, gains[i], Float(slice))
            // w（羽化宽度）在每趟渲染时按频段填，这里先占位
            p.ext = SIMD4<Float>(tanFullH * crop, tanFullV * crop, 0.5 * crop, 0)
            params.append(p)
            slice += 1

            progress(0.48 + 0.34 * Double(i + 1) / Double(count))
        }

        guard !params.isEmpty else { throw StitchError.noInput }

        // 7. 三趟渲染做两频段融合
        //    高频：全分辨率 + 窄羽化（锐、错位只影响几像素）
        //    低频：1/4 分辨率 + 宽羽化（接缝和曝光差被抹平），再把它和「窄羽化的低频」相减，
        //          只把高频细节贴回去。
        //    注：试过把高频改成「每像素只取一帧」的 Voronoi 接缝（Street View 的
        //        seam_finder 思路），但 Tools/verify_seam.py 证明在两频段合成后
        //        会出现白色亮斑、PSNR 反降 1.5dB，需要完整 graphcut 才成立，故未采用。
        let lowW = max(64, outW / 4)
        let lowH = max(32, outH / 4)

        let narrowTex = try renderPass(device: device, pipeline: pipes.stitch,
                                       srcTex: srcTex, base: params,
                                       width: outW, height: outH,
                                       feather: narrowFeather)
        progress(0.62)

        let wideLowTex = try renderPass(device: device, pipeline: pipes.stitch,
                                        srcTex: srcTex, base: params,
                                        width: lowW, height: lowH,
                                        feather: wideFeather)
        let narrowLowTex = try renderPass(device: device, pipeline: pipes.stitch,
                                          srcTex: srcTex, base: params,
                                          width: lowW, height: lowH,
                                          feather: narrowFeather)
        progress(0.80)

        let outTex = try composePass(device: device, pipeline: pipes.compose,
                                     narrow: narrowTex,
                                     wideLow: wideLowTex,
                                     narrowLow: narrowLowTex,
                                     width: outW, height: outH)

        progress(0.86)

        // 9. 读回并统计空洞
        var bytes = [UInt8](repeating: 0, count: outW * outH * 4)
        bytes.withUnsafeMutableBytes { raw in
            if let base = raw.baseAddress {
                outTex.getBytes(base,
                                bytesPerRow: outW * 4,
                                bytesPerImage: outW * 4 * outH,
                                from: MTLRegionMake2D(0, 0, outW, outH),
                                mipmapLevel: 0,
                                slice: 0)
            }
        }
        var holes = 0
        var idx = 3
        let total = outW * outH
        while idx < bytes.count {
            if bytes[idx] == 0 { holes += 1 }
            idx += 4
        }
        let coverage = total > 0 ? Double(total - holes) / Double(total) : 0

        progress(0.94)

        // 10. 生成 CGImage
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let cg = CGImage(width: outW,
                               height: outH,
                               bitsPerComponent: 8,
                               bitsPerPixel: 32,
                               bytesPerRow: outW * 4,
                               space: colorSpace,
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                               provider: provider,
                               decode: nil,
                               shouldInterpolate: true,
                               intent: .defaultIntent) else {
            throw StitchError.encode
        }

        guard let rawJPEG = UIImage(cgImage: cg, scale: 1, orientation: .up)
            .jpegData(compressionQuality: 0.92) else {
            throw StitchError.encode
        }
        // 注入 GPano XMP：不写这段，系统「照片」只会把它当一张普通超宽图
        let jpeg = PhotoSphereMetadata.inject(into: rawJPEG, width: outW, height: outH)

        let thumbnail = makeThumbnail(from: cg)
        progress(1.0)

        return StitchedPanorama(equirect: cg,
                                jpeg: jpeg,
                                thumbnail: thumbnail,
                                coverage: coverage,
                                seconds: Date().timeIntervalSince(start),
                                width: outW,
                                height: outH,
                                residualDeg: residualDeg,
                                pairCount: pairCount,
                                aligner: aligner,
                                calibration: calibrationText)
    }

    // MARK: - 渲染

    /// 用给定分辨率和羽化宽度跑一遍球面投影
    private static func renderPass(device: MTLDevice,
                                   pipeline: MTLComputePipelineState,
                                   srcTex: MTLTexture,
                                   base: [ShaderImage],
                                   width: Int,
                                   height: Int,
                                   feather: Float) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                         width: width,
                                                         height: height,
                                                         mipmapped: false)
        d.usage = [.shaderWrite, .shaderRead]
        d.storageMode = .shared
        guard let dst = device.makeTexture(descriptor: d) else {
            throw StitchError.texture("渲染目标分配失败（\(width)x\(height)）")
        }

        var imgs = base
        for k in 0..<imgs.count { imgs[k].ext.w = feather }

        guard let queue = device.makeCommandQueue(),
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw StitchError.texture("无法创建命令队列")
        }
        enc.setComputePipelineState(pipeline)
        enc.setTexture(srcTex, index: 0)
        enc.setTexture(dst, index: 1)

        var header = Header()
        header.outW = Int32(width)
        header.outH = Int32(height)
        header.count = Int32(imgs.count)
        enc.setBytes(&header, length: MemoryLayout<Header>.stride, index: 0)

        // 参数表用 buffer 而不是 setBytes：setBytes 有 4KB 上限，机位多时会超
        let length = MemoryLayout<ShaderImage>.stride * imgs.count
        let paramBuffer: MTLBuffer? = imgs.withUnsafeBytes { raw in
            guard let p = raw.baseAddress else { return nil }
            return device.makeBuffer(bytes: p, length: length, options: .storageModeShared)
        }
        guard let paramBuffer else { throw StitchError.texture("参数缓冲分配失败") }
        enc.setBuffer(paramBuffer, offset: 0, index: 1)

        let tw = pipeline.threadExecutionWidth
        let th = max(1, pipeline.maxTotalThreadsPerThreadgroup / tw)
        enc.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        if let err = cmd.error {
            throw StitchError.texture(err.localizedDescription)
        }
        return dst
    }

    /// 两频段合成：宽羽化的低频 + 窄羽化的高频
    private static func composePass(device: MTLDevice,
                                    pipeline: MTLComputePipelineState,
                                    narrow: MTLTexture,
                                    wideLow: MTLTexture,
                                    narrowLow: MTLTexture,
                                    width: Int,
                                    height: Int) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                         width: width,
                                                         height: height,
                                                         mipmapped: false)
        d.usage = [.shaderWrite, .shaderRead]
        d.storageMode = .shared
        guard let dst = device.makeTexture(descriptor: d) else {
            throw StitchError.texture("合成目标分配失败")
        }
        guard let queue = device.makeCommandQueue(),
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw StitchError.texture("无法创建命令队列")
        }
        enc.setComputePipelineState(pipeline)
        enc.setTexture(narrow, index: 0)
        enc.setTexture(wideLow, index: 1)
        enc.setTexture(narrowLow, index: 2)
        enc.setTexture(dst, index: 3)

        let tw = pipeline.threadExecutionWidth
        let th = max(1, pipeline.maxTotalThreadsPerThreadgroup / tw)
        enc.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        if let err = cmd.error {
            throw StitchError.texture(err.localizedDescription)
        }
        return dst
    }

    // MARK: - 后备：直接法光度配准

    private struct AlignmentResult {
        var rotations: [simd_float3x3]
        var gains: [Float]
        var residualDeg: Double
        var pairs: Int
        var focalScale: Float
    }

    /// 旧引擎（重叠区光度误差 + 旋转平均），只在特征匹配拿不到足够帧对时启用。
    /// 收敛范围窄（>0.8° 容易掉进局部极小值），但对重复纹理场景是另一套独立证据。
    private static func photometricAlign(_ aligned: [StitchInput],
                                         progress: @escaping (Double) -> Void) -> AlignmentResult? {
        var frames: [FrameAligner.Frame] = []
        frames.reserveCapacity(aligned.count)
        for input in aligned {
            guard let pyr = FrameAligner.pyramid(url: input.url) else { return nil }
            frames.append(FrameAligner.Frame(rotation: input.rotation,
                                             tanH: input.tanHalfH,
                                             tanV: input.tanHalfV,
                                             crop: input.crop,
                                             pyramid: pyr))
        }
        guard frames.count == aligned.count else { return nil }

        let baseTanH = frames.map(\.tanH)
        let baseTanV = frames.map(\.tanV)
        let before = FrameAligner.ghostScore(frames: frames,
                                             rotations: frames.map(\.rotation),
                                             tanH: baseTanH, tanV: baseTanV)

        guard let result = FrameAligner.refine(frames: frames, rounds: 4, progress: progress) else {
            return nil
        }
        let after = FrameAligner.ghostScore(frames: frames,
                                            rotations: result.rotations,
                                            tanH: baseTanH, tanV: baseTanV)

        // 安全阀：规范无关的重叠一致度没有明显改善就整段丢掉
        let enoughPairs = result.pairs >= max(3, frames.count / 3)
        let improved = (before < 0 || after < 0) || after < before * 0.97
        if enoughPairs && improved {
            return AlignmentResult(rotations: result.rotations,
                                   gains: result.gains,
                                   residualDeg: Double(result.residualDeg),
                                   pairs: result.pairs,
                                   focalScale: result.focalScale)
        }
        // 没用上就只借增益（与几何无关，安全）
        return AlignmentResult(rotations: aligned.map(\.rotation),
                               gains: result.gains,
                               residualDeg: -1,
                               pairs: result.pairs,
                               focalScale: 1)
    }

    // MARK: - 辅助

    /// 把所有姿态整体旋转，使第一张画面位于经度 0（等距圆柱图的中间）
    private static func alignToFirst(_ inputs: [StitchInput]) -> [StitchInput] {
        guard let first = inputs.first else { return inputs }
        let f = first.rotation.cameraForward
        let a = rotationZ(-yawOf(f))
        return inputs.map { input in
            StitchInput(url: input.url,
                        rotation: simd_mul(a, input.rotation),
                        tanHalfH: input.tanHalfH,
                        tanHalfV: input.tanHalfV,
                        crop: input.crop,
                        luma: input.luma)
        }
    }

    private static func imagePixelSize(of url: URL) -> CGSize {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Double,
              let h = props[kCGImagePropertyPixelHeight] as? Double else {
            return .zero
        }
        return CGSize(width: w, height: h)
    }

    /// 按目标尺寸解码并重绘为 RGBA8（避免整张全分辨率图驻留内存）
    private static func downsampledRGBA(url: URL, width: Int, height: Int) -> [UInt8]? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: max(width, height)
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else { return nil }

        var data = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        data.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress,
                  let ctx = CGContext(data: base,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: width * 4,
                                      space: colorSpace,
                                      bitmapInfo: info) else { return }
            ctx.interpolationQuality = .high
            // Metal 纹理是顶行优先；CGBitmapContext 默认左下原点，必须翻一次。
            // 这里曾经漏翻，导致 OpenCV 看到的是正帧、Metal 拿到的是倒帧。
            ImageDecoding.drawTopDown(cg, in: ctx, width: width, height: height)
        }
        return data
    }

    /// 从等距圆柱图中央裁一块正常视角的画面做封面
    private static func makeThumbnail(from cg: CGImage) -> Data {
        let w = CGFloat(cg.width)
        let h = CGFloat(cg.height)
        let cropW = min(w, 900)
        let cropH = min(h, cropW * 0.72)
        let rect = CGRect(x: (w - cropW) / 2, y: (h - cropH) / 2, width: cropW, height: cropH)
        let cropped = cg.cropping(to: rect) ?? cg
        let target = UIImage(cgImage: cropped, scale: 1, orientation: .up)
        return target.jpegData(compressionQuality: 0.85) ?? Data()
    }
}
