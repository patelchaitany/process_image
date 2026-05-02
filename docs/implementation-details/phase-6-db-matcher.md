# Phase 6: Face Database & Matcher (SQLite + FAISS) — Implementation Details

This document describes **Phase 6**: persistent **face storage** in **SQLite**, embedding encoding, and the **FaceMatcher** layer that bridges **FAISS-GPU** search with **stable identity IDs**. It aligns with the design spec: [Face Detection & Recognition Pipeline Design](../superpowers/specs/2026-05-02-face-detection-recognition-pipeline-design.md).

**Implementation status note:** **`FaceDatabase`** is **fully implemented** (schema, CRUD paths used by the pipeline, tests). **`FaceMatcher`** currently **loads IDs and names** from the database but **FAISS index build, search, and `add_face` index updates are stubbed** (`TODO` in `face_matcher.cpp`). The sections below document **what the code does today** and the **intended** FAISS/index behavior from the design so Phase 6 can be completed without rediscovering decisions.

---

## What Was Implemented

### Files, classes, and functions

| File | Role |
|------|------|
| `src/matching/face_database.h` | **`FaceRecord`** (`id`, `name`, `embedding`); **`FaceDatabase`** — **`open`**, **`close`**, **`create_tables`**, **`add_face`**, **`load_all`**, **`count`**, **`drop_table`**. |
| `src/matching/face_database.cpp` | SQLite **DDL**, **prepared statements** for insert/select, **BLOB** bind/read for embeddings. |
| `src/matching/face_matcher.h` | **`MatchResult`**; **`FaceMatcher`** — **`init`**, **`release`**, **`match`**, **`add_face`**, **`database_size`**. |
| `src/matching/face_matcher.cpp` | **`init`** loads **`FaceRecord`** list and fills **`ids_`**, **`names_`**; **`match`** / **`add_face`** contain **FAISS TODOs** (placeholder returns unknown). |
| `tests/test_face_database.cpp` | DB lifecycle, multi-row insert, **`drop_table`** re-enroll, **float32 round-trip** accuracy. |

---

## Architecture Decisions

### 1. SQLite schema: `AUTOINCREMENT`, `BLOB`, metadata

**Decision:** Create (if not exists):

```sql
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name);
```

**Reasoning:**

- **`INTEGER PRIMARY KEY AUTOINCREMENT`** gives each enrolled person row a **stable, monotonic `id`** suitable as the canonical **`face_id`** in **`MatchResult`** and in metrics — independent of **FAISS internal indices**.
- **`embedding BLOB`** stores the raw bytes of the float vector — **no text serialization** of floats, compact on disk, exact round-trip for **32-bit** values.
- **`idx_faces_name`** supports enrollment tools and admin queries by **name** without scanning the full table.

### 2. Embedding BLOB encoding: raw `float32`, 2048 bytes per 512-dim face

**Decision:** **`add_face`** binds the blob as:

- Pointer: **`embedding.data()`**
- Byte length: **`embedding.size() * sizeof(float)`**

**`load_all`** reads **`sqlite3_column_blob` / `sqlite3_column_bytes`**, sets **`num_floats = blob_size / sizeof(float)`**, **`memcpy`** into **`std::vector<float>`**.

**Reasoning:**

- **512 × 4 = 2048 bytes** per face for **512-dimensional** embeddings — matches ArcFace output size in the spec.
- Raw IEEE-754 bytes avoid JSON/base64 overhead and preserve **bit-exact** values for **deterministic** FAISS indexing after reload (tests use **`EXPECT_NEAR`** with tight tolerance after round-trip).

### 3. FAISS index selection (intended per design)

**Decision (spec):**

| Database size | Index |
|---------------|--------|
| **< ~1000** faces | **`GpuIndexFlatIP`** — exact inner product on GPU |
| **Larger** | **IVFFlat** (e.g. with training on subset or progressive build) for sublinear search cost |

**Reasoning:**

- **Flat** is simplest and fastest to **build** and **update** for small **N**; latency \< ~0.05 ms for ~1k at 512-d is acceptable per the spec budget table.
- **IVF** trades some recall tuning (**`nprobe`**) for scalability when **N** grows past the low thousands.

### 4. Inner product on L2-normalized vectors = cosine similarity

**Decision (spec + `FaceRecognizer`):** Query and gallery embeddings are **unit L2 norm**. Use **Index with inner product** (e.g. **`IndexFlatIP`**) so **highest IP** = **highest cosine similarity**.

**Reasoning:**

- \(\cos(a,b) = (a \cdot b) / (\|a\| \|b\|)\); when \(\|a\|=\|b\|=1\), **\(\cos = a \cdot b\)**.
- IP is **native** in FAISS; cosine-specialized indexes are unnecessary if inputs are pre-normalized (as **`FaceRecognizer`** already does).

### 5. ID mapping: FAISS row → `ids_` / `names_` (not SQLite row order by accident)

**Decision (intended):** When building the FAISS index from **`load_all()`**, add vectors **in `ORDER BY id`** sequence (already in **`FaceDatabase::load_all`** SQL). **FAISS index position `i`** maps to **`ids_[i]`** and **`names_[i]`**, where **`ids_[i]`** is the **`FaceRecord.id`** (**SQLite `INTEGER PRIMARY KEY`**).

**Reasoning:**

- FAISS returns **integer labels** or **indices** into the added order; that index must map to **application identity**, not to an unstable ordering.
- Using **`FaceRecord.id`** as the **`face_id`** in **`MatchResult`** keeps reporting and optional **SQLite lookups by id** consistent after **add/remove/rebuild** cycles **if** **`ids_` is rebuilt together with the index** from the same **`load_all`** snapshot.

**Current code:** **`FaceMatcher::init`** already builds **`ids_`** and **`names_`** from **`FaceRecord`** order returned by **`ORDER BY id`**, matching this mapping contract once vectors are added to FAISS in the same order.

### 6. Index rebuild vs incremental `add_face`

**Decision (recommended):**

- **Startup:** **`init(db)`** — load all embeddings, **train IVF if needed**, **add** all vectors, store **`ids_` / `names_`**.
- **Enrollment at runtime:** Either **(a)** **`add_with_ids`** / GPU equivalent if the index supports it, or **(b)** for **Flat** small DBs, **rebuild from SQLite** (`load_all` + new index) for **simplicity and correctness** after each enrollment (acceptable when **N** is small and enroll is rare).

**Reasoning:**

- **Rebuild** avoids **index–database drift** if **`add_face` to SQLite** fails after a partial FAISS update (spec: on SQLite failure, index is not updated).
- **IVF** may require **periodic full rebuild** or careful **add** semantics; product choice depends on expected **enrollment frequency** and **N**.

---

## Key Code Snippets

**Schema and index:**

```29:38:src/matching/face_database.cpp
    const char* sql =
        "CREATE TABLE IF NOT EXISTS faces ("
        "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "    name TEXT NOT NULL,"
        "    embedding BLOB NOT NULL,"
        "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ");"
        "CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name);";
```

**BLOB write (raw float bytes):**

```58:60:src/matching/face_database.cpp
    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_blob(stmt, 2, embedding.data(),
                      static_cast<int>(embedding.size() * sizeof(float)), SQLITE_TRANSIENT);
```

**Ordered load for stable FAISS alignment:**

```71:88:src/matching/face_database.cpp
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
```

**Matcher: parallel arrays populated from DB (FAISS TODO):**

```20:34:src/matching/face_matcher.cpp
    auto records = db.load_all();

    ids_.clear();
    names_.clear();

    for (const auto& rec : records) {
        ids_.push_back(rec.id);
        names_.push_back(rec.name);
    }

    // TODO: Phase 6 - Build FAISS index from embeddings
    // For now, store embeddings for brute-force matching
    initialized_ = true;
```

---

## Verification Approach

| Component | Verification |
|-----------|----------------|
| **Schema** | **`FaceDatabaseTest::OpenAndCreateTables`** — open DB, **`count() == 0`**. |
| **Insert / load** | **`AddAndLoadFace`**, **`AddMultipleFaces`** — names, **`embedding.size() == 512`**, element checks. |
| **Round-trip** | **`EmbeddingRoundTrip`** — **`sin`-based** vector, **`EXPECT_NEAR`** per element after SQLite round-trip. |
| **Re-enroll** | **`DropTableAndReenroll`** — **`drop_table`** clears rows, new insert visible. |
| **Matcher (future)** | With FAISS wired: unit test **known** gallery of **normalized** vectors, query = duplicate of row **k**, expect **top-1 id == ids_[k]** and **IP / score** above **`threshold_`**. |
| **GPU parity** | Compare **CPU FAISS-FlatIP** vs **GpuIndexFlatIP** on same small dataset if dual builds are supported. |

---

## Summary

**`FaceDatabase`** implements the spec’s **SQLite** model: **auto-increment IDs**, **`name` + `embedding` BLOB (float32)**, stable **`ORDER BY id`** loading. **`FaceMatcher`** is prepared with **parallel `ids_` / `names_`** aligned to that order; **FAISS GPU index** selection (**FlatIP** vs **IVF**), **search**, and **live index updates** remain to be implemented. Once complete, **inner product** on **L2-normalized** embeddings gives **cosine similarity**, and **FAISS label `i`** should resolve through **`ids_[i]`** to **`sqlite` row id** and display **`names_[i]`**.
