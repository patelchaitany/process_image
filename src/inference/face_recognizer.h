#pragma once

#include "triton_client.h"
#include <vector>
#include <string>

class FaceRecognizer {
public:
    explicit FaceRecognizer(TritonClient& client, const std::string& model_name = "arcface");

    bool recognize(const std::string& input_shm_name,
                   const std::string& output_shm_name,
                   int num_faces,
                   std::vector<std::vector<float>>& embeddings);

    static constexpr int EMBEDDING_DIM = 512;

private:
    TritonClient& client_;
    std::string model_name_;
};
