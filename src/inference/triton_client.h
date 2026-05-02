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
    std::vector<int64_t> outputShape;
    bool isSuccess = false;
    std::string errorMsg;
    float queueTimeNs = 0.0f;
    float computeTimeNs = 0.0f;
    float computeInputTimeNs = 0.0f;
};

/// @brief gRPC client for NVIDIA Triton Inference Server with CUDA shared memory support.
///
/// Connects to Triton at startup, registers CUDA shared memory regions once,
/// and performs zero-copy inference by referencing those regions by name.
class TritonClient {
public:
    TritonClient() = default;
    ~TritonClient();

    TritonClient(const TritonClient&) = delete;
    TritonClient& operator=(const TritonClient&) = delete;

    /// @brief Connect to Triton with retry/backoff.
    /// @param url gRPC endpoint (e.g. "localhost:8001").
    /// @param maxRetries Number of connection attempts before failing.
    /// @param retryDelayMs Milliseconds between retry attempts.
    /// @return true if connection was established successfully.
    bool connect(const std::string& url, int maxRetries = 3, int retryDelayMs = 1000);

    /// @brief Disconnect and unregister all shared memory regions.
    void disconnect();

    /// @brief Check if the client is connected to Triton.
    bool isConnected() const { return isConnected_; }

    /// @brief Register a CUDA device memory region with Triton.
    /// @param name Unique name for this shared memory region.
    /// @param gpuPtr Device pointer to the GPU buffer.
    /// @param byteSize Size of the buffer in bytes.
    /// @param deviceId CUDA device ID (default 0).
    /// @return true if registration succeeded.
    bool registerCudaShm(const std::string& name, void* gpuPtr,
                         size_t byteSize, int deviceId = 0);

    /// @brief Unregister a specific CUDA shared memory region.
    bool unregisterCudaShm(const std::string& name);

    /// @brief Unregister all shared memory regions from Triton.
    void unregisterAllShm();

    /// @brief Run inference on a model using shared memory for I/O.
    /// @param modelName Name of the model in Triton's model repository.
    /// @param inputShmName Name of the registered input shared memory region.
    /// @param inputShape Dimensions of the input tensor.
    /// @param outputShmName Name of the registered output shared memory region.
    /// @param outputShape Expected dimensions of the output tensor.
    /// @return InferResult containing output pointer, shape, and timing stats.
    InferResult infer(const std::string& modelName,
                      const std::string& inputShmName,
                      const std::vector<int64_t>& inputShape,
                      const std::string& outputShmName,
                      const std::vector<int64_t>& outputShape);

private:
    std::string url_;
    bool isConnected_ = false;
    std::unique_ptr<tc::InferenceServerGrpcClient> client_;
    std::unordered_map<std::string, size_t> registeredShm_;

    static constexpr const char* INPUT_TENSOR_NAME = "images";
    static constexpr const char* OUTPUT_TENSOR_NAME = "output0";
};
