#include "cpu_face_matcher.h"
#include <cmath>
#include <algorithm>
#include <cstdio>

bool CPUFaceMatcher::init(FaceDatabase& db, float threshold) {
    threshold_ = threshold;
    records_ = db.load_all();
    initialized_ = true;
    fprintf(stderr, "CPUFaceMatcher: loaded %zu faces\n", records_.size());
    return true;
}

void CPUFaceMatcher::release() {
    records_.clear();
    initialized_ = false;
}

std::vector<MatchResult> CPUFaceMatcher::match(
    const std::vector<std::vector<float>>& embeddings, int k) {

    std::vector<MatchResult> results(embeddings.size());
    (void)k;

    if (!initialized_ || records_.empty()) {
        for (auto& r : results) {
            r.faceId = -1;
            r.confidence = 0.0f;
        }
        return results;
    }

    for (size_t i = 0; i < embeddings.size(); ++i) {
        float best_score = -1.0f;
        int best_idx = -1;

        for (size_t j = 0; j < records_.size(); ++j) {
            float score = cosine_similarity(embeddings[i], records_[j].embedding);
            if (score > best_score) {
                best_score = score;
                best_idx = static_cast<int>(j);
            }
        }

        if (best_score >= threshold_ && best_idx >= 0) {
            results[i].faceId = records_[best_idx].id;
            results[i].name = records_[best_idx].name;
            results[i].confidence = best_score;
        } else {
            results[i].faceId = -1;
            results[i].confidence = 0.0f;
        }
    }

    return results;
}

bool CPUFaceMatcher::add_face(int64_t id, const std::string& name,
                               const std::vector<float>& embedding) {
    FaceRecord rec;
    rec.id = id;
    rec.name = name;
    rec.embedding = embedding;
    records_.push_back(std::move(rec));
    return true;
}

float CPUFaceMatcher::cosine_similarity(const std::vector<float>& a,
                                         const std::vector<float>& b) const {
    if (a.size() != b.size()) return 0.0f;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return (denom > 0) ? dot / denom : 0.0f;
}
