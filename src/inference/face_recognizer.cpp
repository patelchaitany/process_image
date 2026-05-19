#include "face_recognizer.h"
#include <cmath>

FaceRecognizer::FaceRecognizer(TritonClient& client, const std::string& model_name)
    : client_(client), model_name_(model_name) {}

void FaceRecognizer::parseEmbeddings(const InferResult& result, int num_faces,
                                      std::vector<std::vector<float>>& embeddings) {
    embeddings.clear();
    const float* data = result.outputData;
    if (!data) return;

    embeddings.resize(num_faces);
    for (int i = 0; i < num_faces; ++i) {
        embeddings[i].resize(EMBEDDING_DIM);
        const float* emb = data + i * EMBEDDING_DIM;

        float norm = 0.0f;
        for (int j = 0; j < EMBEDDING_DIM; ++j) {
            norm += emb[j] * emb[j];
        }
        norm = std::sqrt(norm);
        for (int j = 0; j < EMBEDDING_DIM; ++j) {
            embeddings[i][j] = (norm > 0.0f) ? emb[j] / norm : 0.0f;
        }
    }
}

bool FaceRecognizer::recognize(const std::string& input_shm_name,
                                const std::string& output_shm_name,
                                int num_faces,
                                std::vector<std::vector<float>>& embeddings,
                                InferResult* outStats) {
    if (num_faces <= 0) return true;

    std::vector<int64_t> inputShape = {num_faces, 3, 112, 112};
    std::vector<int64_t> outputShape = {num_faces, EMBEDDING_DIM};

    InferResult result = client_.infer(model_name_, input_shm_name,
                                        inputShape, output_shm_name, outputShape,
                                        ARCFACE_INPUT_NAME, ARCFACE_OUTPUT_NAME);
    if (outStats) *outStats = result;
    if (!result.isSuccess) return false;

    parseEmbeddings(result, num_faces, embeddings);
    return true;
}

bool FaceRecognizer::recognizeDirect(const float* inputData,
                                      int num_faces,
                                      std::vector<std::vector<float>>& embeddings,
                                      InferResult* outStats) {
    if (num_faces <= 0) return true;

    std::vector<int64_t> inputShape = {num_faces, 3, 112, 112};
    std::vector<int64_t> outputShape = {num_faces, EMBEDDING_DIM};

    InferResult result = client_.inferDirect(model_name_, inputData,
                                              inputShape, outputShape,
                                              ARCFACE_INPUT_NAME, ARCFACE_OUTPUT_NAME);
    if (outStats) *outStats = result;
    if (!result.isSuccess) return false;

    parseEmbeddings(result, num_faces, embeddings);
    return true;
}
