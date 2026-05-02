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
    registeredShm_[name] = {gpuPtr, byteSize};
    fprintf(stderr, "TritonClient: registered CUDA shm '%s' (%zu bytes, device %d)\n",
            name.c_str(), byteSize, deviceId);
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

    size_t inputBytes = sizeof(float) * std::accumulate(
        inputShape.begin(), inputShape.end(), int64_t{1}, std::multiplies<>());
    size_t outputBytes = sizeof(float) * std::accumulate(
        outputShape.begin(), outputShape.end(), int64_t{1}, std::multiplies<>());

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

    tc::InferOptions options(modelName);
    options.model_version_ = "";

    std::vector<tc::InferInput*> inputs = {input.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {output.get()};

    tc::InferResult* rawResult = nullptr;
    err = client_->Infer(&rawResult, options, inputs, outputs);
    if (!err.IsOk()) {
        result.errorMsg = "Inference failed: " + err.Message();
        return result;
    }
    std::unique_ptr<tc::InferResult> inferResult(rawResult);

    auto shmIt = registeredShm_.find(outputShmName);
    if (shmIt == registeredShm_.end() || !shmIt->second.gpuPtr) {
        result.errorMsg = "Output shm region not found or has null GPU pointer";
        return result;
    }

    size_t numFloats = outputBytes / sizeof(float);
    result.ownedOutput.resize(numFloats);
    cudaError_t cudaErr = cudaMemcpy(result.ownedOutput.data(), shmIt->second.gpuPtr,
                                      outputBytes, cudaMemcpyDeviceToHost);
    if (cudaErr != cudaSuccess) {
        result.errorMsg = std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(cudaErr);
        return result;
    }

    result.outputData = result.ownedOutput.data();
    result.outputShape = outputShape;
    result.isSuccess = true;
    return result;
}

InferResult TritonClient::inferDirect(const std::string& modelName,
                                       const float* inputData,
                                       const std::vector<int64_t>& inputShape,
                                       const std::vector<int64_t>& outputShape) {
    InferResult result;
    if (!isConnected_ || !client_) {
        result.errorMsg = "Not connected to Triton";
        return result;
    }

    size_t inputBytes = sizeof(float) * std::accumulate(
        inputShape.begin(), inputShape.end(), int64_t{1}, std::multiplies<>());

    tc::InferInput* rawInput = nullptr;
    tc::Error err = tc::InferInput::Create(&rawInput, INPUT_TENSOR_NAME, inputShape, "FP32");
    if (!err.IsOk()) {
        result.errorMsg = "Failed to create InferInput: " + err.Message();
        return result;
    }
    std::unique_ptr<tc::InferInput> input(rawInput);
    err = input->AppendRaw(reinterpret_cast<const uint8_t*>(inputData), inputBytes);
    if (!err.IsOk()) {
        result.errorMsg = "Failed to append raw input: " + err.Message();
        return result;
    }

    tc::InferRequestedOutput* rawOutput = nullptr;
    err = tc::InferRequestedOutput::Create(&rawOutput, OUTPUT_TENSOR_NAME);
    if (!err.IsOk()) {
        result.errorMsg = "Failed to create InferRequestedOutput: " + err.Message();
        return result;
    }
    std::unique_ptr<tc::InferRequestedOutput> output(rawOutput);

    tc::InferOptions options(modelName);
    options.model_version_ = "";

    std::vector<tc::InferInput*> inputs = {input.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {output.get()};

    tc::InferResult* rawResult = nullptr;
    err = client_->Infer(&rawResult, options, inputs, outputs);
    if (!err.IsOk()) {
        result.errorMsg = "Inference failed: " + err.Message();
        return result;
    }
    std::unique_ptr<tc::InferResult> inferResult(rawResult);

    const uint8_t* outputBuf = nullptr;
    size_t outputBufSize = 0;
    err = inferResult->RawData(OUTPUT_TENSOR_NAME, &outputBuf, &outputBufSize);
    if (!err.IsOk() || !outputBuf) {
        result.errorMsg = "Failed to read output data: " + err.Message();
        return result;
    }

    size_t numFloats = outputBufSize / sizeof(float);
    result.ownedOutput.resize(numFloats);
    std::memcpy(result.ownedOutput.data(), outputBuf, outputBufSize);

    result.outputData = result.ownedOutput.data();
    result.outputShape = outputShape;
    result.isSuccess = true;
    return result;
}
