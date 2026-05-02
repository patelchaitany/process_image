#pragma once

#include "face_database.h"
#include "face_matcher.h"
#include <vector>
#include <string>

class CPUFaceMatcher {
public:
    CPUFaceMatcher() = default;

    bool init(FaceDatabase& db, float threshold = 0.6f);
    void release();

    std::vector<MatchResult> match(const std::vector<std::vector<float>>& embeddings, int k = 1);
    bool add_face(int64_t id, const std::string& name, const std::vector<float>& embedding);
    int database_size() const { return static_cast<int>(records_.size()); }

private:
    float threshold_ = 0.6f;
    std::vector<FaceRecord> records_;
    bool initialized_ = false;

    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) const;
};
