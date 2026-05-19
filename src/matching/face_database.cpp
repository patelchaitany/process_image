#include "face_database.h"
#include <sqlite3.h>
#include <cstdio>
#include <cstring>

FaceDatabase::~FaceDatabase() {
    close();
}

bool FaceDatabase::open(const std::string& db_path) {
    close();
    path_ = db_path;
    int rc = sqlite3_open(db_path.c_str(), reinterpret_cast<sqlite3**>(&db_));
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(static_cast<sqlite3*>(db_)));
        db_ = nullptr;
        return false;
    }
    return create_tables();
}

void FaceDatabase::close() {
    if (db_) {
        sqlite3_close(static_cast<sqlite3*>(db_));
        db_ = nullptr;
    }
}

bool FaceDatabase::create_tables() {
    if (!db_) return false;
    const char* sql =
        "CREATE TABLE IF NOT EXISTS faces ("
        "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "    name TEXT NOT NULL,"
        "    embedding BLOB NOT NULL,"
        "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ");"
        "CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name);";

    char* err_msg = nullptr;
    int rc = sqlite3_exec(static_cast<sqlite3*>(db_), sql, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", err_msg);
        sqlite3_free(err_msg);
        return false;
    }
    return true;
}

bool FaceDatabase::add_face(const std::string& name, const std::vector<float>& embedding,
                            int64_t* out_id) {
    if (!db_) return false;

    const char* sql = "INSERT INTO faces (name, embedding) VALUES (?, ?);";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(static_cast<sqlite3*>(db_), sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return false;

    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_blob(stmt, 2, embedding.data(),
                      static_cast<int>(embedding.size() * sizeof(float)), SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc == SQLITE_DONE && out_id) {
        *out_id = sqlite3_last_insert_rowid(static_cast<sqlite3*>(db_));
    }
    return rc == SQLITE_DONE;
}

std::vector<FaceRecord> FaceDatabase::load_all() {
    std::vector<FaceRecord> records;
    if (!db_) return records;

    const char* sql = "SELECT id, name, embedding FROM faces ORDER BY id;";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(static_cast<sqlite3*>(db_), sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return records;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        FaceRecord rec;
        rec.id = sqlite3_column_int64(stmt, 0);
        rec.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        const void* blob = sqlite3_column_blob(stmt, 2);
        int blob_size = sqlite3_column_bytes(stmt, 2);
        int num_floats = blob_size / sizeof(float);
        rec.embedding.resize(num_floats);
        memcpy(rec.embedding.data(), blob, blob_size);

        records.push_back(std::move(rec));
    }

    sqlite3_finalize(stmt);
    return records;
}

int FaceDatabase::count() {
    if (!db_) return 0;
    const char* sql = "SELECT COUNT(*) FROM faces;";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(static_cast<sqlite3*>(db_), sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return 0;

    int count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return count;
}

bool FaceDatabase::drop_table() {
    if (!db_) return false;
    const char* sql = "DROP TABLE IF EXISTS faces;";
    char* err_msg = nullptr;
    int rc = sqlite3_exec(static_cast<sqlite3*>(db_), sql, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        sqlite3_free(err_msg);
        return false;
    }
    return create_tables();
}
