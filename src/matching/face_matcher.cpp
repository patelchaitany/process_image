#include "face_matcher.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <numeric>

FaceMatcher::~FaceMatcher() {
    release();
}

bool FaceMatcher::init(FaceDatabase& db, float threshold, bool useGpu) {
    threshold_ = threshold;
    useGpu_ = useGpu;
    auto records = db.load_all();
    ids_.clear();
    names_.clear();
    storedEmbeddings_.clear();
    std::vector<float> flatEmbeddings;
    flatEmbeddings.reserve(records.size() * EMBEDDING_DIM);
    for (const auto& rec : records) {
        ids_.push_back(rec.id);
        names_.push_back(rec.name);
        storedEmbeddings_.push_back(rec.embedding);
        flatEmbeddings.insert(flatEmbeddings.end(),
                              rec.embedding.begin(), rec.embedding.end());
    }
    buildIndex(flatEmbeddings, static_cast<int>(records.size()));
    isInitialized_ = true;
    fprintf(stderr, "FaceMatcher: loaded %zu faces (mode: %s)\n",
            ids_.size(), useGpu_ ? "GPU" : "CPU");
    return true;
}

void FaceMatcher::buildIndex(const std::vector<float>& flatEmbeddings, int numFaces) {
    index_.reset();
    gpuResources_.reset();
    if (useGpu_) {
        gpuResources_ = std::make_unique<faiss::gpu::StandardGpuResources>();
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = 0;
        auto gpuIndex = std::make_unique<faiss::gpu::GpuIndexFlatIP>(
            gpuResources_.get(), EMBEDDING_DIM, config);
        if (numFaces > 0) {
            gpuIndex->add(numFaces, flatEmbeddings.data());
        }
        index_ = std::move(gpuIndex);
    } else {
        auto cpuIndex = std::make_unique<faiss::IndexFlatIP>(EMBEDDING_DIM);
        if (numFaces > 0) {
            cpuIndex->add(numFaces, flatEmbeddings.data());
        }
        index_ = std::move(cpuIndex);
    }
}

void FaceMatcher::release() {
    index_.reset();
    gpuResources_.reset();
    ids_.clear();
    names_.clear();
    storedEmbeddings_.clear();
    isInitialized_ = false;
}

std::vector<MatchResult> FaceMatcher::match(
    const std::vector<std::vector<float>>& embeddings, int k) {
    std::vector<MatchResult> results(embeddings.size());
    if (!isInitialized_ || ids_.empty() || !index_) {
        for (auto& r : results) {
            r.faceId = -1;
            r.confidence = 0.0f;
        }
        return results;
    }
    int numQueries = static_cast<int>(embeddings.size());
    // Flatten query embeddings into contiguous buffer
    std::vector<float> queryFlat(numQueries * EMBEDDING_DIM);
    for (int i = 0; i < numQueries; ++i) {
        std::copy(embeddings[i].begin(), embeddings[i].end(),
                  queryFlat.begin() + i * EMBEDDING_DIM);
    }
    // FAISS search: inner product on L2-normalized vectors = cosine similarity
    std::vector<float> distances(numQueries * k);
    std::vector<faiss::idx_t> indices(numQueries * k);
    index_->search(numQueries, queryFlat.data(), k, distances.data(), indices.data());
    // Map FAISS results to MatchResult structs
    for (int i = 0; i < numQueries; ++i) {
        float bestScore = distances[i * k];
        faiss::idx_t bestIdx = indices[i * k];
        if (bestIdx >= 0 && bestIdx < static_cast<faiss::idx_t>(ids_.size()) &&
            bestScore >= threshold_) {
            results[i].faceId = ids_[bestIdx];
            results[i].name = names_[bestIdx];
            results[i].confidence = bestScore;
        } else {
            results[i].faceId = -1;
            results[i].confidence = 0.0f;
        }
    }
    return results;
}

bool FaceMatcher::addFace(int64_t id, const std::string& name,
                           const std::vector<float>& embedding) {
    if (!isInitialized_ || !index_) return false;
    if (static_cast<int>(embedding.size()) != EMBEDDING_DIM) return false;
    index_->add(1, embedding.data());
    ids_.push_back(id);
    names_.push_back(name);
    storedEmbeddings_.push_back(embedding);
    return true;
}
