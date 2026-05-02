#pragma once

#include "face_database.h"
#include <vector>
#include <string>
#include <memory>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

/// @brief Result of matching a single face embedding against the database.
struct MatchResult {
    int64_t faceId = -1;
    std::string name;
    float confidence = 0.0f;
};

/// @brief GPU/CPU face matcher using FAISS for nearest-neighbor search.
///
/// At startup, loads all embeddings from the FaceDatabase, builds a FAISS index
/// (GpuIndexFlatIP for GPU, IndexFlatIP for CPU), and provides cosine similarity
/// matching via inner product on L2-normalized vectors.
class FaceMatcher {
public:
    FaceMatcher() = default;
    ~FaceMatcher();

    FaceMatcher(const FaceMatcher&) = delete;
    FaceMatcher& operator=(const FaceMatcher&) = delete;

    /// @brief Initialize the matcher by loading all faces and building the index.
    /// @param db Reference to the face database.
    /// @param threshold Minimum similarity score to consider a match.
    /// @param useGpu Whether to use GPU-accelerated FAISS index.
    /// @return true if initialization succeeded.
    bool init(FaceDatabase& db, float threshold = 0.6f, bool useGpu = true);

    /// @brief Release all FAISS resources.
    void release();

    /// @brief Match a batch of embeddings against the face database.
    /// @param embeddings Vector of L2-normalized 512-dim embeddings to match.
    /// @param k Number of nearest neighbors to retrieve (default 1).
    /// @return Vector of MatchResult, one per input embedding.
    std::vector<MatchResult> match(const std::vector<std::vector<float>>& embeddings, int k = 1);

    /// @brief Add a new face to the live index without full rebuild.
    /// @param id Database row ID for this face.
    /// @param name Person's name.
    /// @param embedding L2-normalized 512-dim embedding.
    /// @return true if addition succeeded.
    bool addFace(int64_t id, const std::string& name, const std::vector<float>& embedding);

    /// @brief Get the current number of faces in the index.
    int databaseSize() const { return static_cast<int>(ids_.size()); }

private:
    static constexpr int EMBEDDING_DIM = 512;
    static constexpr int IVF_THRESHOLD = 1000;

    float threshold_ = 0.6f;
    bool useGpu_ = true;
    bool isInitialized_ = false;

    std::unique_ptr<faiss::gpu::StandardGpuResources> gpuResources_;
    std::unique_ptr<faiss::Index> index_;

    std::vector<int64_t> ids_;
    std::vector<std::string> names_;
    std::vector<std::vector<float>> storedEmbeddings_;

    void buildIndex(const std::vector<float>& flatEmbeddings, int numFaces);
};
