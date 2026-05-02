#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include "grpc_client.h"

namespace tc = triton::client;

/// @brief Result returned from a single Triton inference call.
struct InferResult {
    const float* outputData = nullptr;
    std::vector<float> ownedOutput;
    std::vector<int64_t> outputShape;
    bool isSuccess = false;
    std::string errorMsg;
    float queueTimeNs = 0.0f;
    float computeTimeNs = 0.0f;
    float computeInputTimeNs = 0.0f;
};

/// @brief Registered shared memory region metadata.
struct ShmRegion {
    void* gpuPtr = nullptr;
    size_t byteSize = 0;
};

/// @brief gRPC client for NVIDIA Triton Inference Server with CUDA shared memory support.
///
/// Supports two inference modes:
/// - GPU path: zero-copy via CUDA shared memory (infer)
/// - CPU path: inline data transfer via gRPC (inferDirect)
class TritonClient {
public:
    TritonClient() = default;
    ~TritonClient();

    TritonClient(const TritonClient&) = delete;
    TritonClient& operator=(const TritonClient&) = delete;

    /// @brief Connect to Triton with retry/backoff.
    bool connect(const std::string& url, int maxRetries = 3, int retryDelayMs = 1000);

    /// @brief Disconnect and unregister all shared memory regions.
    void disconnect();

    /// @brief Check if the client is connected to Triton.
    bool isConnected() const { return isConnected_; }

    /// @brief Register a CUDA device memory region with Triton.
    bool registerCudaShm(const std::string& name, void* gpuPtr,
                         size_t byteSize, int deviceId = 0);

    /// @brief Unregister a specific CUDA shared memory region.
    bool unregisterCudaShm(const std::string& name);

    /// @brief Unregister all shared memory regions from Triton.
    void unregisterAllShm();

    /// @brief Run inference via CUDA shared memory (GPU path).
    /// After Triton writes to the output SHM region, the result is
    /// copied from GPU to InferResult::ownedOutput for CPU-side parsing.
    InferResult infer(const std::string& modelName,
                      const std::string& inputShmName,
                      const std::vector<int64_t>& inputShape,
                      const std::string& outputShmName,
                      const std::vector<int64_t>& outputShape);

    /// @brief Run inference with inline data (CPU path, no shared memory).
    /// Input data is sent via gRPC; output is copied to InferResult::ownedOutput.
    InferResult inferDirect(const std::string& modelName,
                            const float* inputData,
                            const std::vector<int64_t>& inputShape,
                            const std::vector<int64_t>& outputShape);

private:
    std::string url_;
    bool isConnected_ = false;
    std::unique_ptr<tc::InferenceServerGrpcClient> client_;
    std::unordered_map<std::string, ShmRegion> registeredShm_;

    void extractStats(tc::InferResult* inferResult, InferResult& result);

    static constexpr const char* INPUT_TENSOR_NAME = "images";
    static constexpr const char* OUTPUT_TENSOR_NAME = "output0";
};
