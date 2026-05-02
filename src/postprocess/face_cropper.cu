#include "face_cropper.h"
#include <cstdint>
#include <cstdio>

bool FaceCropper::crop_and_preprocess(const void* src_frame_gpu,
                                       const std::vector<Detection>& detections,
                                       void* dst_batch_gpu,
                                       int src_width, int src_height,
                                       cudaStream_t stream) {
    if (detections.empty()) return true;

    // Pack bounding boxes into a flat array for GPU
    std::vector<float> bboxes(detections.size() * 4);
    for (size_t i = 0; i < detections.size(); ++i) {
        bboxes[i * 4 + 0] = detections[i].x1;
        bboxes[i * 4 + 1] = detections[i].y1;
        bboxes[i * 4 + 2] = detections[i].x2;
        bboxes[i * 4 + 3] = detections[i].y2;
    }

    // Upload bboxes to GPU
    float* d_bboxes = nullptr;
    cudaMalloc(&d_bboxes, bboxes.size() * sizeof(float));
    cudaMemcpyAsync(d_bboxes, bboxes.data(), bboxes.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // Use the preprocessor's arcface kernel (defined in preprocessor.cu)
    // Re-using the same kernel logic for face cropping
    const int dst_size = 112;
    dim3 block(16, 16);
    dim3 grid((dst_size + block.x - 1) / block.x,
              (dst_size + block.y - 1) / block.y);

    // The actual kernel is in preprocessor.cu - this calls through that path
    // For now, we directly declare the extern kernel
    extern __global__ void fused_preprocess_arcface_kernel(
        const uint8_t* __restrict__ src,
        float* __restrict__ dst,
        const float* __restrict__ bboxes,
        int src_width, int src_height,
        int face_idx, int dst_size);

    for (size_t i = 0; i < detections.size(); ++i) {
        fused_preprocess_arcface_kernel<<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(src_frame_gpu),
            static_cast<float*>(dst_batch_gpu),
            d_bboxes, src_width, src_height,
            static_cast<int>(i), dst_size);
    }

    cudaFree(d_bboxes);
    return cudaGetLastError() == cudaSuccess;
}
