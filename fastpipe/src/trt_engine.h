#pragma once
// TensorRT engine with batch buckets and CUDA graphs (ARCHITECTURE.md section 5).
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

class TrtEngine {
public:
    // Engine with one optimisation profile; batch is the first input dimension
    TrtEngine(const std::string& plan_path, std::vector<int> buckets, bool use_graphs);
    ~TrtEngine();

    int bucket_for(int n) const;          // smallest bucket >= n (n <= max bucket)
    int bucket_floor(int n) const;        // largest bucket <= n (at least the smallest bucket)
    int max_batch() const { return buckets_.back(); }
    const std::vector<int>& buckets() const { return buckets_; }
    size_t in_item_elems() const { return in_item_elems_; }
    size_t out_item_elems() const { return out_item_elems_; }
    const std::string& name() const { return path_; }

    // Run `bucket` items from `in` to `out` on `s`. The first call for an (in, out, bucket) triple runs eagerly
    // and captures a CUDA graph; later calls replay it. All calls must use the same stream.
    void run(int bucket, void* in, void* out, cudaStream_t s);
    // Capture every bucket's graph for this (in, out) pair now, single-threaded, before the pipeline starts.
    void warm(void* in, void* out, cudaStream_t s);
    // Enqueue without a CUDA graph (for inputs whose address changes every call, e.g. a range of the crop pool)
    void run_direct(int bucket, void* in, void* out, cudaStream_t s);

private:
    nvinfer1::IExecutionContext* ctx_for(int bucket);

    std::string path_;
    std::vector<int> buckets_;
    bool graphs_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::map<int, std::unique_ptr<nvinfer1::IExecutionContext>> ctx_;
    void* scratch_ = nullptr;                 // activation memory shared by every context (one stream)
    std::string in_name_, out_name_;
    nvinfer1::Dims in_dims_{};
    size_t in_item_elems_ = 1, out_item_elems_ = 1;
    std::map<std::tuple<void*, void*, int>, cudaGraphExec_t> graphs_by_key_;
};
