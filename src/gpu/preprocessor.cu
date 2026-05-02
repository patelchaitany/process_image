#include "preprocessor.h"
#include <cstdio>
#include <algorithm>
#include <cmath>

__global__ void fused_preprocess_yolo_kernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int src_width, int src_height,
    int dst_size, float scale, int pad_x, int pad_y)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_size || dy >= dst_size) return;

    float r = 0.0f, g = 0.0f, b = 0.0f;

    int sx = static_cast<int>((dx - pad_x) / scale);
    int sy = static_cast<int>((dy - pad_y) / scale);

    if (sx >= 0 && sx < src_width && sy >= 0 && sy < src_height &&
        dx >= pad_x && dy >= pad_y) {
        int src_idx = (sy * src_width + sx) * 3;
        b = src[src_idx + 0] / 255.0f;
        g = src[src_idx + 1] / 255.0f;
        r = src[src_idx + 2] / 255.0f;
    }

    // CHW layout: R plane, G plane, B plane
    int plane_size = dst_size * dst_size;
    int pixel_idx = dy * dst_size + dx;
    dst[0 * plane_size + pixel_idx] = r;
    dst[1 * plane_size + pixel_idx] = g;
    dst[2 * plane_size + pixel_idx] = b;
}

__global__ void fused_preprocess_arcface_kernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    const float* __restrict__ bboxes,  // [N, 4] as x1,y1,x2,y2
    int src_width, int src_height,
    int face_idx, int dst_size)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_size || dy >= dst_size) return;

    float x1 = bboxes[face_idx * 4 + 0];
    float y1 = bboxes[face_idx * 4 + 1];
    float x2 = bboxes[face_idx * 4 + 2];
    float y2 = bboxes[face_idx * 4 + 3];

    float face_w = x2 - x1;
    float face_h = y2 - y1;

    int sx = static_cast<int>(x1 + dx * face_w / dst_size);
    int sy = static_cast<int>(y1 + dy * face_h / dst_size);

    float r = 0.0f, g = 0.0f, b_val = 0.0f;
    if (sx >= 0 && sx < src_width && sy >= 0 && sy < src_height) {
        int src_idx = (sy * src_width + sx) * 3;
        b_val = (src[src_idx + 0] - 127.5f) / 127.5f;
        g     = (src[src_idx + 1] - 127.5f) / 127.5f;
        r     = (src[src_idx + 2] - 127.5f) / 127.5f;
    }

    int plane_size = dst_size * dst_size;
    int face_offset = face_idx * 3 * plane_size;
    int pixel_idx = dy * dst_size + dx;
    dst[face_offset + 0 * plane_size + pixel_idx] = r;
    dst[face_offset + 1 * plane_size + pixel_idx] = g;
    dst[face_offset + 2 * plane_size + pixel_idx] = b_val;
}

bool Preprocessor::preprocess_yolo(const void* src_bgr_gpu, void* dst_chw_gpu,
                                    int src_width, int src_height, int dst_size,
                                    cudaStream_t stream) {
    float scale_x = static_cast<float>(dst_size) / src_width;
    float scale_y = static_cast<float>(dst_size) / src_height;
    float scale = std::min(scale_x, scale_y);

    int new_w = static_cast<int>(src_width * scale);
    int new_h = static_cast<int>(src_height * scale);
    int pad_x = (dst_size - new_w) / 2;
    int pad_y = (dst_size - new_h) / 2;

    last_params_ = {src_width, src_height, dst_size, scale, pad_x, pad_y};

    // Zero output first (for letterbox padding)
    cudaMemsetAsync(dst_chw_gpu, 0, 3 * dst_size * dst_size * sizeof(float), stream);

    dim3 block(16, 16);
    dim3 grid((dst_size + block.x - 1) / block.x,
              (dst_size + block.y - 1) / block.y);

    fused_preprocess_yolo_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(src_bgr_gpu),
        static_cast<float*>(dst_chw_gpu),
        src_width, src_height, dst_size, scale, pad_x, pad_y);

    return cudaGetLastError() == cudaSuccess;
}

bool Preprocessor::preprocess_arcface(const void* src_bgr_gpu, void* dst_chw_gpu,
                                       const float* bboxes, int num_faces,
                                       int src_width, int src_height,
                                       cudaStream_t stream) {
    const int dst_size = 112;

    dim3 block(16, 16);
    dim3 grid((dst_size + block.x - 1) / block.x,
              (dst_size + block.y - 1) / block.y);

    for (int i = 0; i < num_faces; ++i) {
        fused_preprocess_arcface_kernel<<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(src_bgr_gpu),
            static_cast<float*>(dst_chw_gpu),
            bboxes, src_width, src_height, i, dst_size);
    }

    return cudaGetLastError() == cudaSuccess;
}
