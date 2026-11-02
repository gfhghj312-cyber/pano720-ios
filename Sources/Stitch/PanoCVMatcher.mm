#import "PanoCVMatcher.h"

// OpenCV 5 的模块改名了：features2d -> features，calib3d 拆成 calib / geometry，
// findHomography 与 RANSAC 现在在 geometry/3d.hpp（由 geometry.hpp 聚合）。
#import <opencv2/opencv.hpp>
#import <opencv2/geometry.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

@implementation PanoCVPair
@end

@implementation PanoCVMatcher

+ (BOOL)isAvailable {
    return YES;
}

// MARK: - CGImage -> 灰度 cv::Mat

/// 像素 y 向下、内存第 0 行 = 图像顶行。
/// 这与 Swift 侧 tanHalfV 的映射（ty = (0.5 - v) * 2 * tanV）一致，
/// 一旦这里翻了，解出来的 R 会整体多一个绕 X 的 180°，每帧都会错位。
static bool grayMat(CGImageRef cg, int maxEdge, cv::Mat &out) {
    if (!cg) return false;
    size_t w = CGImageGetWidth(cg);
    size_t h = CGImageGetHeight(cg);
    if (w < 32 || h < 32) return false;

    double scale = 1.0;
    const int longEdge = (int)std::max(w, h);
    if (longEdge > maxEdge) scale = (double)maxEdge / (double)longEdge;

    const int tw = (int)std::lround((double)w * scale);
    const int th = (int)std::lround((double)h * scale);
    if (tw < 32 || th < 32) return false;

    cv::Mat rgba(th, tw, CV_8UC4);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(rgba.data, tw, th, 8, rgba.step[0], cs,
                                             kCGImageAlphaPremultipliedLast |
                                             kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(cs);
    if (!ctx) return false;

    CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
    // CGBitmapContext 原点在左下；先翻转 CTM，让内存第 0 行对应图像顶行
    CGContextTranslateCTM(ctx, 0, th);
    CGContextScaleCTM(ctx, 1, -1);
    CGContextDrawImage(ctx, CGRectMake(0, 0, tw, th), cg);
    CGContextRelease(ctx);

    cv::cvtColor(rgba, out, cv::COLOR_RGBA2GRAY);
    return true;
}

static double patchMean(const cv::Mat &g, const cv::Point2f &p) {
    const int x = (int)std::lround(p.x);
    const int y = (int)std::lround(p.y);
    const int r = 3;
    const int x0 = std::max(0, x - r), x1 = std::min(g.cols, x + r + 1);
    const int y0 = std::max(0, y - r), y1 = std::min(g.rows, y + r + 1);
    if (x1 <= x0 || y1 <= y0) return 0.0;
    return cv::mean(g(cv::Rect(x0, y0, x1 - x0, y1 - y0)))[0];
}

static double medianOf(std::vector<double> &v) {
    if (v.empty()) return 1.0;
    size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    return v[mid];
}

/// cv::Mat(3x3, CV_64F) -> simd_float3x3（列主序）
static simd_float3x3 toSimd(const cv::Mat &R) {
    simd_float3x3 m;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            m.columns[c][r] = (float)R.at<double>(r, c);
        }
    }
    return m;
}

/// 特征检测 + KNN 匹配 + Lowe 比值筛选。任何一步不够就用返回值 false 表示。
template <typename Det>
static bool detectAndMatch(const cv::Ptr<Det> &detector,
                           int normType,
                           double ratio,
                           const cv::Mat &g1,
                           const cv::Mat &g2,
                           std::vector<cv::Point2f> &pa,
                           std::vector<cv::Point2f> &pb,
                           int &rawMatches) {
    if (detector.empty()) return false;

    std::vector<cv::KeyPoint> k1, k2;
    cv::Mat d1, d2;
    detector->detectAndCompute(g1, cv::noArray(), k1, d1);
    detector->detectAndCompute(g2, cv::noArray(), k2, d2);
    if (k1.size() < 40 || k2.size() < 40) return false;
    if (d1.empty() || d2.empty() || d1.type() != d2.type()) return false;

    std::vector<std::vector<cv::DMatch>> knn;
    cv::BFMatcher matcher(normType, false);
    matcher.knnMatch(d1, d2, knn, 2);

    pa.clear();
    pb.clear();
    for (const auto &m : knn) {
        if (m.size() < 2) continue;
        if (m[0].distance >= ratio * m[1].distance) continue;
        pa.push_back(k1[m[0].queryIdx].pt);
        pb.push_back(k2[m[0].trainIdx].pt);
    }
    rawMatches = (int)pa.size();
    return pa.size() >= 24;
}

// MARK: - 入口

+ (nullable PanoCVPair *)matchImageA:(CGImageRef)imageA
                              imageB:(CGImageRef)imageB
                            tanHalfH:(double)tanHalfH
                            tanHalfV:(double)tanHalfV {
    if (tanHalfH < 0.05 || tanHalfV < 0.05) return nil;

    cv::Mat g1, g2;
    if (!grayMat(imageA, 600, g1)) return nil;
    if (!grayMat(imageB, 600, g2)) return nil;

    // 1) 特征。SIFT 是这里的主力：尺度和旋转不变，弱纹理/大角度下比 ORB 稳得多；
    //    纹理实在太少时退回 ORB（对低对比更宽容一点）。
    std::vector<cv::Point2f> pa, pb;
    int rawMatches = 0;
    bool matched = false;
    try {
        matched = detectAndMatch(cv::SIFT::create(1400), cv::NORM_L2, 0.78,
                                 g1, g2, pa, pb, rawMatches);
        if (!matched) {
            matched = detectAndMatch(cv::ORB::create(1800, 1.2f, 8, 31, 0, 2,
                                                     cv::ORB::HARRIS_SCORE, 31, 8),
                                     cv::NORM_HAMMING, 0.80,
                                     g1, g2, pa, pb, rawMatches);
        }
    } catch (const cv::Exception &) {
        return nil;
    }
    if (!matched || pa.size() < 24) return nil;

    // 2) RANSAC 单应。纯旋转时所有深度都满足同一个 H，所以单应模型正好是精确模型
    cv::Mat inlierMask;
    cv::Mat H;
    try {
        H = cv::findHomography(pa, pb, cv::RANSAC, 3.0, inlierMask, 4000, 0.995);
    } catch (const cv::Exception &) {
        return nil;
    }
    if (H.empty() || inlierMask.empty()) return nil;

    const int inliers = cv::countNonZero(inlierMask);
    if (inliers < 24) return nil;
    if ((double)inliers / (double)pa.size() < 0.20) return nil;

    // 3) H = K · R · K⁻¹  →  R = K⁻¹ · H · K，再正交化回 SO(3)
    const double fx = ((double)g1.cols * 0.5) / tanHalfH;
    const double fy = ((double)g1.rows * 0.5) / tanHalfV;
    const double cx = (double)g1.cols * 0.5;
    const double cy = (double)g1.rows * 0.5;

    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                                            0, fy, cy,
                                            0, 0, 1);
    cv::Mat Rraw = K.inv() * H * K;

    cv::Mat U, w, Vt;
    try {
        cv::SVD::compute(Rraw, w, U, Vt, cv::SVD::FULL_UV);
    } catch (const cv::Exception &) {
        return nil;
    }
    cv::Mat R = U * Vt;
    if (cv::determinant(R) < 0) {
        cv::Mat U2 = U.clone();
        U2.col(2) *= -1;
        R = U2 * Vt;
    }

    // 4) 换回本工程的相机约定：像素 y 向下 → 相机 y 向上
    //    S = diag(1, -1, -1)，R_ours = S · R_cv · S
    cv::Mat S = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                                            0, -1, 0,
                                            0, 0, -1);
    cv::Mat Rours = S * R * S;

    // 5) 合理性：纯旋转不可能一步转太多
    double tr = Rours.at<double>(0, 0) + Rours.at<double>(1, 1) + Rours.at<double>(2, 2);
    double cosA = std::max(-1.0, std::min(1.0, (tr - 1.0) * 0.5));
    const double angleDeg = std::acos(cosA) * 180.0 / 3.14159265358979323846;
    if (angleDeg > 45.0) return nil;

    // 6) 亮度比（供逐帧增益用）
    std::vector<double> ratios;
    ratios.reserve(inliers);
    const uchar *mask = inlierMask.ptr<uchar>(0);
    for (size_t i = 0; i < pa.size(); ++i) {
        if (mask[i] == 0) continue;
        const double a = patchMean(g1, pa[i]);
        const double b = patchMean(g2, pb[i]);
        if (a > 6.0 && b > 6.0) ratios.push_back(b / a);
    }
    double ratioGain = 1.0;
    if (!ratios.empty()) {
        std::vector<double> logs;
        logs.reserve(ratios.size());
        for (double r : ratios) logs.push_back(std::log(std::max(1e-3, r)));
        ratioGain = std::exp(medianOf(logs));
    }
    ratioGain = std::max(0.4, std::min(2.5, ratioGain));

    PanoCVPair *out = [PanoCVPair new];
    out.rotation = toSimd(Rours);
    out.inliers = (NSInteger)inliers;
    out.rawMatches = (NSInteger)rawMatches;
    out.brightnessRatio = ratioGain;
    return out;
}

@end
