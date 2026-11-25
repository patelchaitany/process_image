#include "trt_engine.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <tuple>

#include "common.h"

namespace {
class Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::fprintf(stderr, "[TRT] %s\n", msg);
    }
} g_logger;
}  // namespace

TrtEngine::TrtEngine(const std::string& plan_path, std::vector<int> buckets, bool use_graphs)
    : path_(plan_path), buckets_(std::move(buckets)), graphs_(use_graphs) {
    std::ifstream f(plan_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open engine " + plan_path);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    runtime_.reset(nvinfer1::createInferRuntime(g_logger));
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) throw std::runtime_error("cannot deserialize " + plan_path);
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* n = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(n) == nvinfer1::TensorIOMode::kINPUT) in_name_ = n; else out_name_ = n;
    }
    in_dims_ = engine_->getProfileShape(in_name_.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
    for (int d = 1; d < in_dims_.nbDims; ++d) in_item_elems_ *= in_dims_.d[d];
    std::sort(buckets_.begin(), buckets_.end());
    const int maxb = in_dims_.d[0];
    buckets_.erase(std::remove_if(buckets_.begin(), buckets_.end(), [&](int b) { return b > maxb; }), buckets_.end());
    if (buckets_.empty() || buckets_.back() != maxb) buckets_.push_back(maxb);
    CK(cudaMalloc(&scratch_, engine_->getDeviceMemorySizeV2()));
    // output size per item, from a context at the largest bucket
    auto* c = ctx_for(buckets_.back());
    nvinfer1::Dims od = c->getTensorShape(out_name_.c_str());
    for (int d = 1; d < od.nbDims; ++d) out_item_elems_ *= od.d[d];
    std::fprintf(stderr, "[engine] %s in=%s out=%s item=%zu->%zu buckets=%zu scratch=%.1f MB graphs=%d\n",
                 plan_path.c_str(), in_name_.c_str(), out_name_.c_str(), in_item_elems_, out_item_elems_,
                 buckets_.size(), engine_->getDeviceMemorySizeV2() / 1e6, (int)graphs_);
}

TrtEngine::~TrtEngine() {
    for (auto& kv : graphs_by_key_) cudaGraphExecDestroy(kv.second);
    ctx_.clear();
    if (scratch_) cudaFree(scratch_);
}

int TrtEngine::bucket_for(int n) const {
    for (int b : buckets_) if (b >= n) return b;
    return buckets_.back();
}

int TrtEngine::bucket_floor(int n) const {
    int b = buckets_.front();
    for (int x : buckets_) if (x <= n) b = x;
    return b;
}

nvinfer1::IExecutionContext* TrtEngine::ctx_for(int bucket) {
    auto it = ctx_.find(bucket);
    if (it != ctx_.end()) return it->second.get();
    auto* c = engine_->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);
    c->setDeviceMemoryV2(scratch_, engine_->getDeviceMemorySizeV2());
    nvinfer1::Dims d = in_dims_;
    d.d[0] = bucket;
    if (!c->setInputShape(in_name_.c_str(), d)) throw std::runtime_error("bad input shape");
    ctx_[bucket].reset(c);
    return c;
}

void TrtEngine::run(int bucket, void* in, void* out, cudaStream_t s) {
    auto* c = ctx_for(bucket);
    if (!graphs_) {
        c->setTensorAddress(in_name_.c_str(), in);
        c->setTensorAddress(out_name_.c_str(), out);
        if (!c->enqueueV3(s)) throw std::runtime_error("enqueueV3 failed");
        return;
    }
    auto key = std::make_tuple(in, out, bucket);
    auto it = graphs_by_key_.find(key);
    if (it == graphs_by_key_.end()) {
        c->setTensorAddress(in_name_.c_str(), in);
        c->setTensorAddress(out_name_.c_str(), out);
        if (!c->enqueueV3(s)) throw std::runtime_error("enqueueV3 failed");   // flush deferred work before capture
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        if (!c->enqueueV3(s)) throw std::runtime_error("enqueueV3 (capture) failed");
        CK(cudaStreamEndCapture(s, &g));
        cudaGraphExec_t ge;
        CK(cudaGraphInstantiate(&ge, g, 0));
        CK(cudaGraphDestroy(g));
        graphs_by_key_[key] = ge;
        return;    // the eager run above already produced this batch's output
    }
    CK(cudaGraphLaunch(it->second, s));
}

void TrtEngine::warm(void* in, void* out, cudaStream_t s) {
    for (int b : buckets_) run(b, in, out, s);
    CK(cudaStreamSynchronize(s));
}

void TrtEngine::run_direct(int bucket, void* in, void* out, cudaStream_t s) {
    auto* c = ctx_for(bucket);
    c->setTensorAddress(in_name_.c_str(), in);
    c->setTensorAddress(out_name_.c_str(), out);
    if (!c->enqueueV3(s)) throw std::runtime_error("enqueueV3 failed");
}
