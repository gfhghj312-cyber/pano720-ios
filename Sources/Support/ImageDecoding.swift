import Foundation
import CoreGraphics

/// 所有「CGImage → 原始像素缓冲区」都必须走同一个方向约定。
///
/// `CGImage` 是顶行优先，而 `CGBitmapContext` 默认左下原点；直接 `draw` 得到的
/// 内存行序与图像相反。Metal 纹理却是顶行优先（纹理坐标 (0,0) = 第一字节）。
/// 忘了翻转时，OpenCV 看到的是正帧、Metal 拿到的是倒帧，拼接会出现大面积半透明重影。
enum ImageDecoding {

    /// 在指定的 bitmap context 中按「内存第 0 行 = 图像顶行」绘制。
    static func drawTopDown(_ image: CGImage,
                            in context: CGContext,
                            width: Int,
                            height: Int) {
        context.saveGState()
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        context.restoreGState()
    }
}
