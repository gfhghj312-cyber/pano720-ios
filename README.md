# 全景 720 · iPhone 720° 全景相机

手持转动手机即可拍摄 **360°×180° 球面全景**（俗称 720°）的 iOS App。
拍摄时给出球面机位指引、自动对齐、自动快门；拍完在设备上用 Metal 直接拼接成等距圆柱全景图。

面向 iOS 26 / 27 的 **Liquid Glass** 视觉语言，用 SwiftUI 写成，不依赖任何第三方库。

---

## 功能

| 模块 | 说明 |
| --- | --- |
| 球面机位网格 | 贪心球面覆盖：按真实 FOV 和视场矩形精确判定，构造上保证 100% 覆盖、不留空洞 |
| 镜头切换 | 取景器左上角 `0.5×` / `1×`，默认 1×。0.5× 超广角一圈约 **14 张**，1× 约 **35 张**；切换失败会在提示条写明原因，不会静默 |
| 实时指引 | 中央准星 + 目标机位标记 + 视野外方向箭头 + 「向右转 32°」文字提示 |
| 小地图 | 球面展开图，已拍机位实心、未拍空心，十字为当前朝向 |
| 自动快门 | 目标机位角误差 < 2.2° 且 0.4s 内抖动 < 0.9° 时自动拍摄 |
| 手动模式 | 关闭自动后可手动按快门，未对准也允许拍（像素依然参与拼接） |
| 姿态对齐 | 按视频帧 PTS 在姿态历史里做四元数插值，补偿视频流 1~2 帧延迟 |
| 姿态对齐 | OpenCV SIFT/ORB 特征 + RANSAC 单应解相对旋转，再做全局旋转平均，把陀螺仪 1~3° 的漂移修回来 |
| 拼接 | Metal compute shader：逐像素求世界射线 → 反投影到每帧采样 → cos³ 加权混合 |
| 融合 | 两频段：宽羽化的低频 + 窄羽化的高频，接缝看不见、重影限制在几像素内 |
| 补洞 | 第二轮对未覆盖像素放宽边界做边缘外推，杜绝黑块 |
| 曝光校正 | 用重叠区亮度比解逐帧增益（0.70–1.45），比全局均值回归准得多 |
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
# 先准备好 OpenCV 静态框架（CI 会自动下载并缓存，本地要手动来一次）
mkdir -p Vendor
curl -L -o /tmp/opencv.zip https://github.com/opencv/opencv/releases/download/5.0.0/opencv-5.0.0-ios-framework.zip
ditto -x -k /tmp/opencv.zip Vendor      # 必须用 ditto，unzip 会把框架里的符号链接拆坏

xcodegen generate
open Pano720.xcodeproj     # 真机调试需要自己选 Team 签名
```

> OpenCV 是静态框架（arm64 设备 + x86_64 模拟器），链接后二进制约 6.8 MB。
> 注意 OpenCV 5 改了模块名（`features2d`→`features`、`calib3d` 拆成 `calib`/`geometry`），
> `findHomography` 在 `geometry.hpp` 里，AKAZE 已被移除 —— 这份代码按 5.0 写。

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

## 拼接原理

分两段：**先修姿态（OpenCV 特征）**，再用姿态做球面重投影 + 两频段融合。

### 第一段：姿态从哪来

每张照片都记录了 CoreMotion 的姿态四元数（`.xArbitraryZVertical` 参考系，Z 轴垂直），
但陀螺仪拍一整圈会漂 1~3°，直接用会把同一根门框画两遍（就是肉眼看到的半透明重影）。
所以真机上一律先跑一次特征对齐：

```
每帧与视场最近的 4 个邻居：SIFT（弱纹理退回 ORB）→ 交叉校验/比值筛选 → RANSAC 单应 H
纯旋转模型下 H = K·R·K⁻¹ 与深度无关，所以 R = K⁻¹HK 是精确解（再 SVD 正交化）
把「Rⱼᵀ Rᵢ = Mᵢⱼ」当约束，装配 3N×3N 正规方程解出每帧修正量（旋转平均），12 轮
```

**为什么这一层要交给 OpenCV**：自己写直接法光度配准的收敛范围只有 0.8°，
且对曝光变化和重复纹理极敏感；特征匹配是这一类问题的成熟解法，SIFT 对尺度/旋转/光照都稳。
OpenCV 只用来算「相对旋转」这一件事，球面投影和融合仍然自己控制（输出严格 2:1 等距圆柱）。

**阻尼是必须的**：环状机位图的结构矩阵有一类「整体缓慢旋转」的弱约束方向，接近奇异。
不压住的话修正量会逐轮指数放大（实测 3.6→2.3→2.6→3.1→4.1→5.4→7.2°），3 轮后整圈姿态飞出 23°。
`Tools/verify_rotation_averaging.py` 把这个坑固化了，含发散对照组。

失败链：特征对齐（与陀螺仪差 >25° 就放弃）→ 光度配准 → 纯陀螺仪。结果页会显示实际走的哪条。

### 第二段：重投影

有了姿态，每帧在球面上的位置就确定了：

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

### 两频段融合（接缝和重影的关键）

单频段羽化只能在两个坏结果里二选一：羽化窄了接缝硬、羽化宽了错位被摊成大片半透明重影。
所以渲染三趟再合成：

| 层 | 分辨率 | 羽化 | 作用 |
| --- | --- | --- | --- |
| 高频 | 全分辨率 | 0.055 | 保留锐度，错位只影响几像素 |
| 低频 | 1/4 | 0.30 | 接缝和曝光差被抹平 |
| 低频参照 | 1/4 | 0.055 | 用来把高频从窄羽化结果里分离出来 |

`out = 低频(宽) + [高频(窄) − 低频(窄)]`，也就是把宽羽化的低频和窄羽化的高频拼起来。

整张 4096×2048 一次 compute dispatch 完成。所有源帧打包成一个 `texture2d_array`，
单次绑定，避免 30 多个纹理参数；源图降采样尺寸按张数动态计算，显存占用控制在 ~170MB。

拼接正确与否几乎全取决于 `R` 的方向。`CMAttitude.rotationMatrix` 到底是「设备→参考系」还是
「参考系→设备」，Apple 文档与坊间代码互相矛盾；猜错一次就是全糊。所以这里不猜：
参考系里重力恒为 `(0,0,-1)`，而 `deviceMotion.gravity` 是设备系下的重力，
两者一比就能判出是否该转置（`MotionService.start()`），之后全链路统一用「相机→世界」，着色器里再取 `Rᵀ`。

## 调参位置

| 想要的效果 | 改哪里 |
| --- | --- |
| 张数更少 / 更快 | `PanoramaGrid.nodes(overlap:)` 调小（0.10 更激进，仍保证 0 空洞） |
| 拼接更顺滑 | 同上调大（0.24） |
| 输出更高分辨率 | `CaptureModel.stitch()` 里 `outWidth: 4096` → 8192（内存翻 4 倍） |
| 自动快门更灵敏 | `CaptureModel.handle()` 里 `alignmentDeg < 2.2` 与 `0.016`（抖动阈值） |
| 羽化范围 | `StitchService.narrowFeather` / `wideFeather`（0.055 / 0.30） |
| 配准轮数 | `FeatureAligner.align()` 的 `for _ in 0..<12`（少于 8 收敛不干净） |
| 阻尼强度 | `FeatureAligner` 里 `A[k*dim+k] += 0.6`（调到 1e-3 会发散，见验证脚本） |
| 每帧匹配邻居数 | `FeatureAligner.align(maxNeighbors:)`（4；调大更稳但更慢） |
| 特征图尺寸 | `PanoCVMatcher.mm` 里 `grayMat(..., 600, ...)`（调大更准但更慢） |
| 超广角裁切比例 | `CameraLens.usableCrop`（0.92；调到 1.0 用满整块传感器，边角会变软） |
| 曝光校正强度 | `FrameAligner.solveGains` 里的钳制区间 `0.70...1.45` |

## 数学验证

`Tools/verify_stitch.py` 是这套投影的闭环仿真：合成一个已知全景 → 按机位反投影出帧（模拟拍摄）
→ 用与 Metal 着色器逐字对应的公式拼回 → 和真值比。跑法：

```
pip install numpy pillow && python Tools/verify_stitch.py
```

实测结果（1280×640 输出）：

| 情形 | 空洞 | PSNR |
| --- | --- | --- |
| 正确传 Rᵀ（当前实现） | 0.000% | 31.4 dB |
| 传 R（旧代码的 bug） | 37% | 7.9 dB |

误差图和真值逐像素比是均匀的，只剩等距圆柱极区固有的 1~2 行拉伸 —— 没有任何接缝条带。
加上 ±0.7° 的随机姿态误差后 PSNR 几乎不变，说明混合权重对几度以内的漂移不敏感。

## 已知限制

- 精配准解的是「旋转」，不建稀疏点云、不做完整 bundle adjustment，也不处理视差（机位平移）。
  所以贴得很近的物体（30cm 内的桌角）在重叠区仍可能错开 —— 转手机的时候尽量原地转，别走位。
- 结果页会显示「对齐残差」。正常应该在 0.2~0.5°；如果经常 >1°，说明拍摄时转得太快或者光线太暗。
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
  Stitch/StitchService.swift    Metal 拼接内核 + 调度
  Stitch/FeatureAligner.swift   OpenCV 特征对齐 + 旋转平均 + 逐帧增益（主引擎）
  Stitch/PanoCVMatcher.mm       OpenCV 封装（SIFT/ORB + RANSAC 单应 → 相对旋转）
  Stitch/FrameAligner.swift     直接法光度配准（后备引擎）
  Views/                        界面
.github/workflows/build-ipa.yml 免签打包
Tools/make_icon.py              程序化生成 App 图标
```
