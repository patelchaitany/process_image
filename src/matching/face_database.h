#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct FaceRecord {
    int64_t id;
    std::string name;
    std::vector<float> embedding;  // 512-dim
};

class FaceDatabase {
public:
    FaceDatabase() = default;
    ~FaceDatabase();

    bool open(const std::string& db_path);
    void close();
    bool is_open() const { return db_ != nullptr; }

    bool create_tables();
    bool add_face(const std::string& name, const std::vector<float>& embedding);
    std::vector<FaceRecord> load_all();
    int count();

    bool drop_table();

private:
    void* db_ = nullptr;  // sqlite3*
    std::string path_;
};
