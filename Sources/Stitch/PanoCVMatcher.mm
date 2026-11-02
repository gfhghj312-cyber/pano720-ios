#import "PanoCVMatcher.h"

#import <opencv2/opencv.hpp>

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

// MARK: - 入口

+ (nullable PanoCVPair *)matchImageA:(CGImageRef)imageA
                              imageB:(CGImageRef)imageB
                            tanHalfH:(double)tanHalfH
                            tanHalfV:(double)tanHalfV {
    if (tanHalfH < 0.05 || tanHalfV < 0.05) return nil;

    cv::Mat g1, g2;
    if (!grayMat(imageA, 720, g1)) return nil;
    if (!grayMat(imageB, 720, g2)) return nil;

    // 1) 特征：ORB 足够快；纹理太少时退回 AKAZE（更抗尺度/低对比）
    std::vector<cv::KeyPoint> k1, k2;
    cv::Mat d1, d2;
    try {
        cv::Ptr<cv::ORB> orb = cv::ORB::create(1600, 1.2f, 8, 31, 0, 2,
                                               cv::ORB::HARRIS_SCORE, 31, 12);
        orb->detectAndCompute(g1, cv::noArray(), k1, d1);
        orb->detectAndCompute(g2, cv::noArray(), k2, d2);

        if (k1.size() < 80 || k2.size() < 80) {
            cv::Ptr<cv::AKAZE> ak = cv::AKAZE::create();
            ak->detectAndCompute(g1, cv::noArray(), k1, d1);
            ak->detectAndCompute(g2, cv::noArray(), k2, d2);
        }
    } catch (const cv::Exception &) {
        return nil;
    }

    if (k1.size() < 20 || k2.size() < 20) return nil;
    if (d1.type() != d2.type()) return nil;

    // 2) 交叉校验匹配（都是一次性比完，交叉校验能干掉大半误匹配）
    std::vector<cv::DMatch> matches;
    try {
        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        matcher.match(d1, d2, matches);
    } catch (const cv::Exception &) {
        return nil;
    }
    if (matches.size() < 24) return nil;

    std::vector<cv::Point2f> pa, pb;
    pa.reserve(matches.size());
    pb.reserve(matches.size());
    for (const auto &m : matches) {
        pa.push_back(k1[m.queryIdx].pt);
        pb.push_back(k2[m.trainIdx].pt);
    }

    // 3) RANSAC 单应。纯旋转时所有深度都满足同一个 H，所以单应模型正好
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
    if ((double)inliers / (double)matches.size() < 0.20) return nil;

    // 4) H = K · R · K⁻¹  →  R = K⁻¹ · H · K，再正交化回 SO(3)
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

    // 5) 换回本工程的相机约定：像素 y 向下 → 相机 y 向上
    //    S = diag(1, -1, -1)，R_ours = S · R_cv · S
    cv::Mat S = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                                            0, -1, 0,
                                            0, 0, -1);
    cv::Mat Rours = S * R * S;

    // 6) 合理性：纯旋转不可能一步转太多；SVD 的中间奇异值必须接近 1
    double tr = Rours.at<double>(0, 0) + Rours.at<double>(1, 1) + Rours.at<double>(2, 2);
    double cosA = std::max(-1.0, std::min(1.0, (tr - 1.0) * 0.5));
    const double angleDeg = std::acos(cosA) * 180.0 / 3.14159265358979323846;
    if (angleDeg > 45.0) return nil;

    // 7) 亮度比（供逐帧增益用）
    std::vector<double> ratios;
    ratios.reserve(inliers);
    const uchar *mask = inlierMask.ptr<uchar>(0);
    for (size_t i = 0; i < pa.size(); ++i) {
        if (mask[i] == 0) continue;
        const double a = patchMean(g1, pa[i]);
        const double b = patchMean(g2, pb[i]);
        if (a > 6.0 && b > 6.0) ratios.push_back(b / a);
    }
    double ratio = 1.0;
    if (!ratios.empty()) {
        std::vector<double> logs;
        logs.reserve(ratios.size());
        for (double r : ratios) logs.push_back(std::log(std::max(1e-3, r)));
        ratio = std::exp(medianOf(logs));
    }
    ratio = std::max(0.4, std::min(2.5, ratio));

    PanoCVPair *out = [PanoCVPair new];
    out.rotation = toSimd(Rours);
    out.inliers = (NSInteger)inliers;
    out.rawMatches = (NSInteger)matches.size();
    out.brightnessRatio = ratio;
    return out;
}

@end
