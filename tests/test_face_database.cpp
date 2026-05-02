#include <gtest/gtest.h>
#include "matching/face_database.h"
#include <filesystem>
#include <cmath>

class FaceDatabaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        db_path_ = "/tmp/test_faces_" + std::to_string(getpid()) + ".db";
        db_.open(db_path_);
    }

    void TearDown() override {
        db_.close();
        std::filesystem::remove(db_path_);
    }

    std::string db_path_;
    FaceDatabase db_;
};

TEST_F(FaceDatabaseTest, OpenAndCreateTables) {
    EXPECT_TRUE(db_.is_open());
    EXPECT_EQ(db_.count(), 0);
}

TEST_F(FaceDatabaseTest, AddAndLoadFace) {
    std::vector<float> embedding(512, 0.0f);
    embedding[0] = 1.0f;

    EXPECT_TRUE(db_.add_face("alice", embedding));
    EXPECT_EQ(db_.count(), 1);

    auto records = db_.load_all();
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(records[0].name, "alice");
    EXPECT_EQ(records[0].embedding.size(), 512);
    EXPECT_FLOAT_EQ(records[0].embedding[0], 1.0f);
    EXPECT_FLOAT_EQ(records[0].embedding[1], 0.0f);
}

TEST_F(FaceDatabaseTest, AddMultipleFaces) {
    std::vector<float> emb1(512, 0.1f);
    std::vector<float> emb2(512, 0.2f);
    std::vector<float> emb3(512, 0.3f);

    EXPECT_TRUE(db_.add_face("alice", emb1));
    EXPECT_TRUE(db_.add_face("bob", emb2));
    EXPECT_TRUE(db_.add_face("charlie", emb3));
    EXPECT_EQ(db_.count(), 3);

    auto records = db_.load_all();
    ASSERT_EQ(records.size(), 3);
    EXPECT_EQ(records[0].name, "alice");
    EXPECT_EQ(records[1].name, "bob");
    EXPECT_EQ(records[2].name, "charlie");
}

TEST_F(FaceDatabaseTest, DropTableAndReenroll) {
    std::vector<float> emb(512, 0.5f);
    db_.add_face("alice", emb);
    EXPECT_EQ(db_.count(), 1);

    EXPECT_TRUE(db_.drop_table());
    EXPECT_EQ(db_.count(), 0);

    db_.add_face("bob", emb);
    EXPECT_EQ(db_.count(), 1);
    auto records = db_.load_all();
    EXPECT_EQ(records[0].name, "bob");
}

TEST_F(FaceDatabaseTest, EmbeddingRoundTrip) {
    std::vector<float> embedding(512);
    for (int i = 0; i < 512; ++i) {
        embedding[i] = std::sin(static_cast<float>(i) * 0.1f);
    }

    db_.add_face("test", embedding);
    auto records = db_.load_all();
    ASSERT_EQ(records.size(), 1);

    for (int i = 0; i < 512; ++i) {
        EXPECT_NEAR(records[0].embedding[i], embedding[i], 1e-6f)
            << "Mismatch at index " << i;
    }
}
