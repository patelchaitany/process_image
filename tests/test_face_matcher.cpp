#include <gtest/gtest.h>
#include "matching/face_database.h"
#include "matching/face_matcher.h"
#include <filesystem>
#include <cmath>
#include <random>

class FaceMatcherTest : public ::testing::Test {
protected:
    void SetUp() override {
        dbPath_ = "/tmp/test_matcher_" + std::to_string(getpid()) + ".db";
        db_.open(dbPath_);
    }

    void TearDown() override {
        db_.close();
        std::filesystem::remove(dbPath_);
    }

    std::vector<float> makeNormalizedEmbedding(int seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> emb(512);
        float norm = 0.0f;
        for (auto& v : emb) {
            v = dist(rng);
            norm += v * v;
        }
        norm = std::sqrt(norm);
        for (auto& v : emb) v /= norm;
        return emb;
    }

    std::string dbPath_;
    FaceDatabase db_;
};

TEST_F(FaceMatcherTest, InitWithEmptyDatabase) {
    FaceMatcher matcher;
    EXPECT_TRUE(matcher.init(db_, 0.6f, false));
    EXPECT_EQ(matcher.databaseSize(), 0);
}

TEST_F(FaceMatcherTest, MatchAgainstEmptyDatabase) {
    FaceMatcher matcher;
    matcher.init(db_, 0.6f, false);
    auto emb = makeNormalizedEmbedding(42);
    auto results = matcher.match({emb});
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].faceId, -1);
    EXPECT_FLOAT_EQ(results[0].confidence, 0.0f);
}

TEST_F(FaceMatcherTest, ExactMatchReturnsHighConfidence) {
    auto emb = makeNormalizedEmbedding(100);
    db_.add_face("alice", emb);
    FaceMatcher matcher;
    matcher.init(db_, 0.6f, false);
    EXPECT_EQ(matcher.databaseSize(), 1);
    auto results = matcher.match({emb});
    ASSERT_EQ(results.size(), 1);
    EXPECT_GT(results[0].faceId, 0);
    EXPECT_EQ(results[0].name, "alice");
    EXPECT_GT(results[0].confidence, 0.99f);
}

TEST_F(FaceMatcherTest, DifferentEmbeddingNoMatch) {
    auto emb1 = makeNormalizedEmbedding(1);
    auto emb2 = makeNormalizedEmbedding(9999);
    db_.add_face("alice", emb1);
    FaceMatcher matcher;
    matcher.init(db_, 0.8f, false);
    auto results = matcher.match({emb2});
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].faceId, -1);
}

TEST_F(FaceMatcherTest, MultipleQueriesBatchMatch) {
    auto embAlice = makeNormalizedEmbedding(10);
    auto embBob = makeNormalizedEmbedding(20);
    db_.add_face("alice", embAlice);
    db_.add_face("bob", embBob);
    FaceMatcher matcher;
    matcher.init(db_, 0.6f, false);
    auto results = matcher.match({embAlice, embBob});
    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].name, "alice");
    EXPECT_EQ(results[1].name, "bob");
}

TEST_F(FaceMatcherTest, AddFaceIncrementally) {
    FaceMatcher matcher;
    matcher.init(db_, 0.6f, false);
    EXPECT_EQ(matcher.databaseSize(), 0);
    auto emb = makeNormalizedEmbedding(77);
    EXPECT_TRUE(matcher.addFace(1, "dave", emb));
    EXPECT_EQ(matcher.databaseSize(), 1);
    auto results = matcher.match({emb});
    EXPECT_EQ(results[0].name, "dave");
    EXPECT_GT(results[0].confidence, 0.99f);
}
