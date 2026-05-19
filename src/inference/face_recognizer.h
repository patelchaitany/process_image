#pragma once

#include "triton_client.h"
#include <vector>
#include <string>

class FaceRecognizer {
public:
    explicit FaceRecognizer(TritonClient& client, const std::string& model_name = "arcface");

    /// @brief Run recognition via CUDA shared memory (GPU path).
    bool recognize(const std::string& input_shm_name,
                   const std::string& output_shm_name,
                   int num_faces,
                   std::vector<std::vector<float>>& embeddings,
                   InferResult* outStats = nullptr);

    /// @brief Run recognition with inline data (CPU path).
    bool recognizeDirect(const float* inputData,
                         int num_faces,
                         std::vector<std::vector<float>>& embeddings,
                         InferResult* outStats = nullptr);

    static constexpr int EMBEDDING_DIM = 512;
    static constexpr const char* ARCFACE_INPUT_NAME = "input";
    static constexpr const char* ARCFACE_OUTPUT_NAME = "output";

private:
    TritonClient& client_;
    std::string model_name_;

    void parseEmbeddings(const InferResult& result, int num_faces,
                         std::vector<std::vector<float>>& embeddings);
};
