#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <simd/simd.h>

NS_ASSUME_NONNULL_BEGIN

/// 一帧相对另一帧的纯旋转估计（OpenCV 特征匹配 + RANSAC 单应分解）
@interface PanoCVPair : NSObject

/// 把相机 A 的射线转到相机 B 的旋转。
/// 约定与 Swift 侧一致：前 = -Z，右 = +X，上 = +Y（即 ray = (x, y, -1) 归一化）。
/// 数值上等于 R_Bᵀ · R_A（R 为「相机 -> 世界」）。
@property (nonatomic, assign) simd_float3x3 rotation;

/// RANSAC 内点数
@property (nonatomic, assign) NSInteger inliers;

/// 交叉校验后的原始匹配数
@property (nonatomic, assign) NSInteger rawMatches;

/// B 相对 A 的亮度比（> 1 表示 B 更亮），由内点小块的灰度均值中位数得到
@property (nonatomic, assign) double brightnessRatio;

@end

/// OpenCV 特征匹配封装。OpenCV 是 C++，Swift 无法直接调用，
/// 这一层只暴露纯 ObjC 接口（不泄漏任何 cv:: 类型）。
@interface PanoCVMatcher : NSObject

+ (BOOL)isAvailable;

/// 估计 A -> B 的相对旋转。返回 nil 表示这一对不可用
/// （纹理不足 / 重叠不足 / 几何不一致 / 解出的旋转不像纯旋转）。
+ (nullable PanoCVPair *)matchImageA:(CGImageRef)imageA
                              imageB:(CGImageRef)imageB
                            tanHalfH:(double)tanHalfH
                            tanHalfV:(double)tanHalfV;

@end

NS_ASSUME_NONNULL_END
