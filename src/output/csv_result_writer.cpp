#include "output/csv_result_writer.h"
#include "postprocess/nms.h"
#include "matching/face_matcher.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <cstdio>

CsvResultWriter::CsvResultWriter(const std::string& path,
                                   int rotateSizeMb,
                                   int flushIntervalMs)
    : basePath_(path),
      rotateSizeBytes_(rotateSizeMb * 1024 * 1024),
      flushIntervalMs_(flushIntervalMs) {}

CsvResultWriter::~CsvResultWriter() {
    close();
}

bool CsvResultWriter::open() {
    auto dir = std::filesystem::path(basePath_).parent_path();
    if (!dir.empty()) {
        std::filesystem::create_directories(dir);
    }

    file_.open(basePath_, std::ios::out | std::ios::app);
    if (!file_.is_open()) {
        fprintf(stderr, "CsvResultWriter: cannot open %s\n", basePath_.c_str());
        return false;
    }

    if (std::filesystem::file_size(basePath_) == 0) {
        writeHeader();
    } else {
        headerWritten_ = true;
    }

    running_ = true;
    writerThread_ = std::thread(&CsvResultWriter::writerLoop, this);
    return true;
}

void CsvResultWriter::write(const FrameResult& result) {
    if (!running_.load(std::memory_order_relaxed)) return;
    buffer_.push(result);
}

void CsvResultWriter::writerLoop() {
    while (running_.load(std::memory_order_relaxed)) {
        FrameResult entry;
        bool wrote = false;

        while (buffer_.pop(entry)) {
            writeEntry(entry);
            wrote = true;
        }

        if (wrote) {
            file_.flush();
            rotateIfNeeded();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(flushIntervalMs_));
    }
}

void CsvResultWriter::drain() {
    FrameResult entry;
    while (buffer_.pop(entry)) {
        writeEntry(entry);
    }
    if (file_.is_open()) {
        file_.flush();
    }
}

void CsvResultWriter::close() {
    if (running_.exchange(false)) {
        if (writerThread_.joinable()) {
            writerThread_.join();
        }
        drain();
    }
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

void CsvResultWriter::writeHeader() {
    file_ << "frame_id,timestamp_utc,source_id,det_idx,"
          << "x1,y1,x2,y2,bbox_width,bbox_height,det_confidence,"
          << "identity,face_id,match_confidence\n";
    headerWritten_ = true;
}

void CsvResultWriter::writeEntry(const FrameResult& result) {
    if (!file_.is_open()) return;

    if (!headerWritten_) {
        writeHeader();
    }

    if (result.detections.empty()) {
        file_ << result.frame_id << ','
              << result.timestamp_utc << ','
              << result.source_id << ','
              << 0 << ','
              << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ','
              << 0 << ',' << 0 << ','
              << 0 << ','
              << "no_detection" << ',' << -1 << ',' << 0 << '\n';
        return;
    }

    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& det = result.detections[i];
        float bw = det.x2 - det.x1;
        float bh = det.y2 - det.y1;

        std::string identity = "unknown";
        int64_t faceId = -1;
        float matchConf = 0.0f;

        if (i < result.matches.size() && result.matches[i].faceId >= 0) {
            identity = result.matches[i].name;
            faceId = result.matches[i].faceId;
            matchConf = result.matches[i].confidence;
        }

        file_ << result.frame_id << ','
              << result.timestamp_utc << ','
              << result.source_id << ','
              << i << ','
              << std::fixed << std::setprecision(1)
              << det.x1 << ',' << det.y1 << ','
              << det.x2 << ',' << det.y2 << ','
              << bw << ',' << bh << ','
              << std::setprecision(4) << det.confidence << ','
              << identity << ',' << faceId << ','
              << matchConf << '\n';
    }
}

void CsvResultWriter::rotateIfNeeded() {
    if (rotateSizeBytes_ <= 0) return;

    auto pos = static_cast<int>(file_.tellp());
    if (pos >= rotateSizeBytes_) {
        file_.close();
        std::string rotated = generateRotatedPath();
        std::filesystem::rename(basePath_, rotated);
        file_.open(basePath_, std::ios::out);
        headerWritten_ = false;
        writeHeader();
    }
}

std::string CsvResultWriter::generateRotatedPath() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&time, &tm);

    auto stem = std::filesystem::path(basePath_).stem().string();
    auto dir = std::filesystem::path(basePath_).parent_path().string();
    auto ext = std::filesystem::path(basePath_).extension().string();

    std::ostringstream ss;
    ss << dir << '/' << stem << '_'
       << std::put_time(&tm, "%Y%m%d_%H%M%S") << ext;
    return ss.str();
}
