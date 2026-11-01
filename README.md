# 全景 720 · iPhone 720° 全景相机

手持转动手机即可拍摄 **360°×180° 球面全景**（俗称 720°）的 iOS App。
拍摄时给出球面机位指引、自动对齐、自动快门；拍完在设备上用 Metal 直接拼接成等距圆柱全景图。

面向 iOS 26 / 27 的 **Liquid Glass** 视觉语言，用 SwiftUI 写成，不依赖任何第三方库。

---

## 功能

| 模块 | 说明 |
| --- | --- |
| 球面机位网格 | 按后置相机真实视场角（FOV）自动计算环数与每环机位数，默认 28% 重叠 |
| 实时指引 | 中央准星 + 目标机位标记 + 视野外方向箭头 + 「向右转 32°」文字提示 |
| 小地图 | 球面展开图，已拍机位实心、未拍空心，十字为当前朝向 |
| 自动快门 | 目标机位角误差 < 2.2° 且 0.4s 内抖动 < 0.9° 时自动拍摄 |
| 手动模式 | 关闭自动后可手动按快门，未对准也允许拍（像素依然参与拼接） |
| 拼接 | Metal compute shader：逐像素求世界射线 → 反投影到每帧采样 → 边缘羽化 + 视角余弦加权混合 |
| 曝光校正 | 每帧亮度均值向全局均值回归（增益 0.72–1.38），抑制自动曝光跳变 |
| 输出 | 4096×2048 等距圆柱 JPEG，可存相册 / 分享 / 在 App 内拖动查看 |
| 相册 | 本地保存，缩略图取全景中央裁剪的「正常视角」画面 |

## 快速开始（不需要 Mac）

1. 代码推到 GitHub（本仓库已配置好）。
2. `Actions` → `Build unsigned IPA` → 等 3~5 分钟。
3. 下载 artifact `Pano720-unsigned-ipa`，解压得到 `Pano720-unsigned.ipa`。
4. 用你的自签证书（Sideloadly / AltStore / SignTools / TrollStore）签名安装。

工作流做的事情：`xcodegen` 生成工程 → `xcodebuild CODE_SIGNING_ALLOWED=NO` 编译 → 把 `.app` 塞进 `Payload/` 打成 zip 改名 `.ipa`。**全程不签名**，签名字段完全交给你。

### 在 Mac 上手动构建

```bash
brew install xcodegen
xcodegen generate
open Pano720.xcodeproj     # 真机调试需要自己选 Team 签名
```

## 自签注意点

- Bundle ID 是 `com.pano720.app`，可随意改成你自己的（`project.yml` 里的 `PRODUCT_BUNDLE_IDENTIFIER`）。
- 免费 Apple ID 自签 7 天到期；TrollStore 可永久。
- 首次启动需要授权 **相机**（必须）与 **相册写入**（存图时才会问）。
- 若对着相机黑屏：确认装在真机上（模拟器没有摄像头）。

## 拍摄技巧

1. 站定，脚别动，先对准正前方按第一张。
2. 沿着提示往一个方向匀速转身体（像 Google 街景那样），每次停半秒。
3. 天顶那张需要把手机竖起来对准天花板；天底（脚下）是可选的，不拍也能出图。
4. 光线均匀的环境效果最好；逆光/高反差时拼接处会有轻微色差（当前只做了亮度增益校正）。

## 拼接原理（为什么不用 OpenCV）

每张照片都记录了 CoreMotion 的**姿态四元数**（`.xArbitraryZVertical` 参考系，Z 轴垂直），
因此每帧在球面上的位置是已知的，不需要做特征点匹配：

```
对输出等距圆柱图的每个像素 (u,v)：
  lon = (u-0.5)·2π,  lat = (0.5-v)·π
  w   = (cos lat·cos lon, cos lat·sin lon, sin lat)     // 世界射线
  for 每一帧 i:
      c = Rᵢᵀ · w                                        // 转到相机系
      if c.z >= 0: continue                              // 在背后
      (su, sv) = 针孔投影(c, tanHalfH, tanHalfV)
      wgt = 边缘羽化(su,sv) · cos³(偏离光轴角) · 增益ᵢ
      累加 color·wgt 与 wgt
  输出 = Σcolor·wgt / Σwgt   （权重为 0 的像素标记为空洞）
```

整张 4096×2048 一次 compute dispatch 完成。所有源帧打包成一个 `texture2d_array`，
单次绑定，避免 30 多个纹理参数；源图降采样尺寸按张数动态计算，显存占用控制在 ~140MB。

## 调参位置

| 想要的效果 | 改哪里 |
| --- | --- |
| 张数更少 / 更快 | `PanoramaGrid.nodes(overlap:)` 调小（0.20 更激进） |
| 拼接更顺滑 | 同上调大（0.35） |
| 输出更高分辨率 | `CaptureModel.stitch()` 里 `outWidth: 4096` → 8192（内存翻 4 倍） |
| 自动快门更灵敏 | `CaptureModel.handle()` 里 `alignmentDeg < 2.2` 与 `0.016`（抖动阈值） |
| 羽化范围 | `StitchService` 着色器里 `smoothstep(0.0, 0.17, eu)` |
| 曝光校正强度 | 着色器 `gain` 的钳制区间 `0.72...1.38` |

## 已知限制

- 纯姿态拼接，不做特征点二次优化：手机陀螺仪漂移大时，接缝处会有轻微错位（正常使用下约 1~2 像素）。
- 运动物体（行人、车）在重叠区会出现重影，这是所有曝光式全景的共性。
- 只做了亮度增益，没做颜色/多波段融合（multi-band blending）。
- 输出是标准等距圆柱 JPEG，但没写 GPano XMP 元数据，所以系统「照片」不会自动识别为全景球。
  App 内有自己的查看器，可以拖动/缩放无缝浏览。

## 目录结构

```
project.yml                     XcodeGen 工程定义（CI 用它生成 .xcodeproj）
Config/Info.plist               权限与方向（锁定竖屏，保证投影数学一致）
Sources/
  App/Pano720App.swift
  Support/Math.swift            球面坐标 / 四元数工具
  Support/Theme.swift           Liquid Glass 封装（iOS 26+ 原生，低版本材质回退）
  Models/PanoramaStore.swift    本地相册存储（JSON + 文件）
  Capture/PanoramaGrid.swift    机位网格生成
  Capture/MotionService.swift   CoreMotion 姿态
  Capture/CameraController.swift AVFoundation 采集（视频流，逐帧已知朝向）
  Capture/CaptureModel.swift    状态机：对齐判定 / 自动快门 / 拼接调度
  Stitch/StitchService.swift    Metal 拼接内核
  Views/                        界面
.github/workflows/build-ipa.yml 免签打包
Tools/make_icon.py              程序化生成 App 图标
```
