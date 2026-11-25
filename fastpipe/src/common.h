#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CK(x)                                                                                          \
    do {                                                                                               \
        cudaError_t e_ = (x);                                                                          \
        if (e_ != cudaSuccess) {                                                                       \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,               \
                         cudaGetErrorString(e_));                                                      \
            std::abort();                                                                              \
        }                                                                                              \
    } while (0)

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// NVTX range for Nsight Systems timelines
struct Nvtx {
    explicit Nvtx(const char* name) { nvtxRangePushA(name); }
    ~Nvtx() { nvtxRangePop(); }
};

// Name the calling OS thread in Nsight Systems
inline void name_thread(const char* name) { nvtxNameOsThreadA((uint32_t)syscall(SYS_gettid), name); }
