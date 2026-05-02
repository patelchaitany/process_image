#pragma once

#include <chrono>
#include <cuda_runtime.h>

class CpuTimer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    void stop() { end_ = std::chrono::high_resolution_clock::now(); }

    float elapsed_ms() const {
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - start_);
        return duration.count() / 1e6f;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_;
};

class ScopedCpuTimer {
public:
    explicit ScopedCpuTimer(float& out_ms) : out_(out_ms) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    ~ScopedCpuTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_);
        out_ = duration.count() / 1e6f;
    }

private:
    float& out_;
    std::chrono::high_resolution_clock::time_point start_;
};

class CudaEventTimer {
public:
    CudaEventTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~CudaEventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void record_start(cudaStream_t stream = nullptr) {
        cudaEventRecord(start_, stream);
    }

    void record_stop(cudaStream_t stream = nullptr) {
        cudaEventRecord(stop_, stream);
    }

    float elapsed_ms() const {
        float ms = 0.0f;
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

    float elapsed_ms_async() const {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

    cudaEvent_t start_event() const { return start_; }
    cudaEvent_t stop_event() const { return stop_; }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};
