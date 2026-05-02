#include "cpu_preprocessor.h"
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#endif

bool CPUPreprocessor::preprocess_yolo(const uint8_t* src_bgr, float* dst_chw,
                                       int src_width, int src_height, int dst_size) {
    float scale_x = static_cast<float>(dst_size) / src_width;
    float scale_y = static_cast<float>(dst_size) / src_height;
    float scale = std::min(scale_x, scale_y);

    int new_w = static_cast<int>(src_width * scale);
    int new_h = static_cast<int>(src_height * scale);
    int pad_x = (dst_size - new_w) / 2;
    int pad_y = (dst_size - new_h) / 2;

    last_params_ = {src_width, src_height, dst_size, scale, pad_x, pad_y};

#ifdef HAS_OPENCV
    cv::Mat src(src_height, src_width, CV_8UC3, const_cast<uint8_t*>(src_bgr));
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat canvas = cv::Mat::zeros(dst_size, dst_size, CV_8UC3);
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));

    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

    cv::Mat blob = cv::dnn::blobFromImage(rgb, 1.0 / 255.0, cv::Size(), cv::Scalar(), false, false);
    std::memcpy(dst_chw, blob.data, 3 * dst_size * dst_size * sizeof(float));
#else
    // Fallback: manual bilinear resize (no OpenCV)
    std::memset(dst_chw, 0, 3 * dst_size * dst_size * sizeof(float));

    for (int dy = 0; dy < dst_size; ++dy) {
        for (int dx = 0; dx < dst_size; ++dx) {
            int sx = static_cast<int>((dx - pad_x) / scale);
            int sy = static_cast<int>((dy - pad_y) / scale);

            if (sx < 0 || sx >= src_width || sy < 0 || sy >= src_height ||
                dx < pad_x || dy < pad_y) continue;

            int src_idx = (sy * src_width + sx) * 3;
            int pixel_idx = dy * dst_size + dx;
            int plane_size = dst_size * dst_size;

            // BGR to RGB + normalize
            dst_chw[0 * plane_size + pixel_idx] = src_bgr[src_idx + 2] / 255.0f;  // R
            dst_chw[1 * plane_size + pixel_idx] = src_bgr[src_idx + 1] / 255.0f;  // G
            dst_chw[2 * plane_size + pixel_idx] = src_bgr[src_idx + 0] / 255.0f;  // B
        }
    }
#endif

    return true;
}

bool CPUPreprocessor::preprocess_arcface(const uint8_t* src_bgr, float* dst_chw,
                                          const float* bboxes, int num_faces,
                                          int src_width, int src_height) {
    const int dst_size = 112;
    int plane_size = dst_size * dst_size;

    for (int fi = 0; fi < num_faces; ++fi) {
        float x1 = bboxes[fi * 4 + 0];
        float y1 = bboxes[fi * 4 + 1];
        float x2 = bboxes[fi * 4 + 2];
        float y2 = bboxes[fi * 4 + 3];
        float face_w = x2 - x1;
        float face_h = y2 - y1;

        int face_offset = fi * 3 * plane_size;

#ifdef HAS_OPENCV
        int ix1 = std::max(0, static_cast<int>(x1));
        int iy1 = std::max(0, static_cast<int>(y1));
        int ix2 = std::min(src_width, static_cast<int>(x2));
        int iy2 = std::min(src_height, static_cast<int>(y2));

        cv::Mat src(src_height, src_width, CV_8UC3, const_cast<uint8_t*>(src_bgr));
        cv::Mat crop = src(cv::Rect(ix1, iy1, ix2 - ix1, iy2 - iy1));
        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(dst_size, dst_size));

        for (int dy = 0; dy < dst_size; ++dy) {
            for (int dx = 0; dx < dst_size; ++dx) {
                auto pixel = resized.at<cv::Vec3b>(dy, dx);
                int idx = dy * dst_size + dx;
                dst_chw[face_offset + 0 * plane_size + idx] = (pixel[2] - 127.5f) / 127.5f;
                dst_chw[face_offset + 1 * plane_size + idx] = (pixel[1] - 127.5f) / 127.5f;
                dst_chw[face_offset + 2 * plane_size + idx] = (pixel[0] - 127.5f) / 127.5f;
            }
        }
#else
        for (int dy = 0; dy < dst_size; ++dy) {
            for (int dx = 0; dx < dst_size; ++dx) {
                int sx = static_cast<int>(x1 + dx * face_w / dst_size);
                int sy = static_cast<int>(y1 + dy * face_h / dst_size);

                float r = 0, g = 0, b = 0;
                if (sx >= 0 && sx < src_width && sy >= 0 && sy < src_height) {
                    int src_idx = (sy * src_width + sx) * 3;
                    b = (src_bgr[src_idx + 0] - 127.5f) / 127.5f;
                    g = (src_bgr[src_idx + 1] - 127.5f) / 127.5f;
                    r = (src_bgr[src_idx + 2] - 127.5f) / 127.5f;
                }

                int idx = dy * dst_size + dx;
                dst_chw[face_offset + 0 * plane_size + idx] = r;
                dst_chw[face_offset + 1 * plane_size + idx] = g;
                dst_chw[face_offset + 2 * plane_size + idx] = b;
            }
        }
#endif
    }

    return true;
}
