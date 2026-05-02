#include "triton_client.h"
#include <cstdio>
#include <thread>
#include <chrono>
#include <numeric>
#include <cuda_runtime.h>

TritonClient::~TritonClient() {
    disconnect();
}

bool TritonClient::connect(const std::string& url, int maxRetries, int retryDelayMs) {
    url_ = url;
    for (int attempt = 1; attempt <= maxRetries; ++attempt) {
        fprintf(stderr, "TritonClient: connecting to %s (attempt %d/%d)...\n",
                url.c_str(), attempt, maxRetries);
        tc::Error err = tc::InferenceServerGrpcClient::Create(&client_, url);
        if (!err.IsOk()) {
            fprintf(stderr, "TritonClient: create failed: %s\n", err.Message().c_str());
            if (attempt < maxRetries) {
                std::this_thread::sleep_for(std::chrono::milliseconds(retryDelayMs));
            }
            continue;
        }
        bool isLive = false;
        err = client_->IsServerLive(&isLive);
        if (err.IsOk() && isLive) {
            isConnected_ = true;
            fprintf(stderr, "TritonClient: connected successfully\n");
            return true;
        }
        fprintf(stderr, "TritonClient: server not live\n");
        if (attempt < maxRetries) {
            std::this_thread::sleep_for(std::chrono::milliseconds(retryDelayMs));
        }
    }
    fprintf(stderr, "TritonClient: FAILED to connect after %d attempts\n", maxRetries);
    return false;
}

void TritonClient::disconnect() {
    unregisterAllShm();
    isConnected_ = false;
    client_.reset();
}

bool TritonClient::registerCudaShm(const std::string& name, void* gpuPtr,
                                    size_t byteSize, int deviceId) {
    if (!isConnected_) return false;
    cudaIpcMemHandle_t handle;
    cudaError_t cudaErr = cudaIpcGetMemHandle(&handle, gpuPtr);
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "TritonClient: cudaIpcGetMemHandle failed: %s\n",
                cudaGetErrorString(cudaErr));
        return false;
    }
    tc::Error err = client_->RegisterCudaSharedMemory(
        name, handle, deviceId, byteSize);
    if (!err.IsOk()) {
        fprintf(stderr, "TritonClient: register CUDA shm '%s' failed: %s\n",
                name.c_str(), err.Message().c_str());
        return false;
    }
    registeredShm_[name] = byteSize;
    fprintf(stderr, "TritonClient: registered CUDA shm '%s' (%zu bytes)\n",
            name.c_str(), byteSize);
    return true;
}

bool TritonClient::unregisterCudaShm(const std::string& name) {
    if (!isConnected_) return false;
    tc::Error err = client_->UnregisterCudaSharedMemory(name);
    if (!err.IsOk()) {
        fprintf(stderr, "TritonClient: unregister '%s' failed: %s\n",
                name.c_str(), err.Message().c_str());
        return false;
    }
    registeredShm_.erase(name);
    return true;
}

void TritonClient::unregisterAllShm() {
    if (!isConnected_ || !client_) return;
    tc::Error err = client_->UnregisterCudaSharedMemory("");
    if (err.IsOk()) {
        registeredShm_.clear();
    }
}

InferResult TritonClient::infer(const std::string& modelName,
                                 const std::string& inputShmName,
                                 const std::vector<int64_t>& inputShape,
                                 const std::string& outputShmName,
                                 const std::vector<int64_t>& outputShape) {
    InferResult result;
    if (!isConnected_ || !client_) {
        result.errorMsg = "Not connected to Triton";
        return result;
    }
    // Compute byte sizes from shapes
    size_t inputBytes = sizeof(float) * std::accumulate(
        inputShape.begin(), inputShape.end(), int64_t{1}, std::multiplies<>());
    size_t outputBytes = sizeof(float) * std::accumulate(
        outputShape.begin(), outputShape.end(), int64_t{1}, std::multiplies<>());
    // Create input tensor referencing CUDA shared memory
    tc::InferInput* rawInput = nullptr;
    tc::Error err = tc::InferInput::Create(&rawInput, INPUT_TENSOR_NAME, inputShape, "FP32");
    if (!err.IsOk()) {
        result.errorMsg = "Failed to create InferInput: " + err.Message();
        return result;
    }
    std::unique_ptr<tc::InferInput> input(rawInput);
    err = input->SetSharedMemory(inputShmName, inputBytes);
    if (!err.IsOk()) {
        result.errorMsg = "Failed to set input shm: " + err.Message();
        return result;
    }
    // Create output tensor referencing CUDA shared memory
    tc::InferRequestedOutput* rawOutput = nullptr;
    err = tc::InferRequestedOutput::Create(&rawOutput, OUTPUT_TENSOR_NAME);
    if (!err.IsOk()) {
        result.errorMsg = "Failed to create InferRequestedOutput: " + err.Message();
        return result;
    }
    std::unique_ptr<tc::InferRequestedOutput> output(rawOutput);
    err = output->SetSharedMemory(outputShmName, outputBytes);
    if (!err.IsOk()) {
        result.errorMsg = "Failed to set output shm: " + err.Message();
        return result;
    }
    // Configure inference options
    tc::InferOptions options(modelName);
    options.model_version_ = "";
    // Execute inference
    std::vector<tc::InferInput*> inputs = {input.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {output.get()};
    tc::InferResult* rawResult = nullptr;
    err = client_->Infer(&rawResult, options, inputs, outputs);
    if (!err.IsOk()) {
        result.errorMsg = "Inference failed: " + err.Message();
        return result;
    }
    std::unique_ptr<tc::InferResult> inferResult(rawResult);
    // Extract timing statistics from the response
    std::string modelStat;
    inferResult->ModelName(&modelStat);
    tc::InferStat stat;
    if (inferResult->Statistics(&stat).IsOk()) {
        result.queueTimeNs = static_cast<float>(stat.queue_ns);
        result.computeTimeNs = static_cast<float>(stat.compute_infer_ns);
        result.computeInputTimeNs = static_cast<float>(stat.compute_input_ns);
    }
    // Output data lives in the CUDA shared memory region -- retrieve pointer
    // The caller owns the memory (it's in their pre-registered GPU buffer)
    auto shmIt = registeredShm_.find(outputShmName);
    if (shmIt == registeredShm_.end()) {
        result.errorMsg = "Output shm region not found in registry";
        return result;
    }
    result.isSuccess = true;
    result.outputShape = outputShape;
    // outputData will be set by caller who owns the GPU pointer
    result.outputData = nullptr;
    return result;
}
