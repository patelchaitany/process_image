#include "output/webhook_result_writer.h"
#include "matching/face_matcher.h"
#include "postprocess/nms.h"
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace {

std::mutex gCurlMutex;
int gCurlUserCount = 0;

void curlGlobalAddRef() {
    std::lock_guard<std::mutex> lock(gCurlMutex);
    if (gCurlUserCount == 0) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    }
    ++gCurlUserCount;
}

void curlGlobalReleaseRef() {
    std::lock_guard<std::mutex> lock(gCurlMutex);
    --gCurlUserCount;
    if (gCurlUserCount == 0) {
        curl_global_cleanup();
    }
}

std::string escapeJsonString(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20U) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }
    return out;
}

void appendMatchObject(std::ostringstream& oss, const MatchResult& m) {
    oss << std::fixed << std::setprecision(7);
    oss << "{\"person_id\":" << m.faceId << ",\"name\":\"" << escapeJsonString(m.name)
        << "\",\"similarity\":" << m.confidence << '}';
}

void appendDetectionObject(std::ostringstream& oss, const FrameResult& result, size_t idx) {
    const Detection& d = result.detections[idx];
    oss << std::fixed << std::setprecision(7);
    oss << "{\"bbox\":{\"x1\":" << d.x1 << ",\"y1\":" << d.y1 << ",\"x2\":" << d.x2
        << ",\"y2\":" << d.y2 << "},\"confidence\":" << d.confidence << ",\"match\":";
    if (idx < result.matches.size() && result.matches[idx].faceId != -1) {
        appendMatchObject(oss, result.matches[idx]);
    } else {
        oss << "null";
    }
    oss << '}';
}

void appendDetectionsArray(std::ostringstream& oss, const FrameResult& result) {
    oss << "\"detections\":[";
    for (size_t i = 0; i < result.detections.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        appendDetectionObject(oss, result, i);
    }
    oss << ']';
}

} // namespace

WebhookResultWriter::WebhookResultWriter(std::string callback_url, std::string session_id)
    : callbackUrl_(std::move(callback_url)),
      sessionId_(std::move(session_id)) {}

WebhookResultWriter::~WebhookResultWriter() {
    close();
}

bool WebhookResultWriter::open() {
    bool shouldStartThread = false;
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        if (senderThread_.joinable()) {
            return curl_ != nullptr;
        }
        curlGlobalAddRef();
        curl_.reset(curl_easy_init());
        if (!curl_) {
            fprintf(stderr, "WebhookResultWriter: curl_easy_init failed\n");
            curlGlobalReleaseRef();
            return false;
        }
        running_.store(true, std::memory_order_release);
        shouldStartThread = true;
    }
    if (shouldStartThread) {
        senderThread_ = std::thread(&WebhookResultWriter::senderLoop, this);
    }
    return true;
}

void WebhookResultWriter::write(const FrameResult& result) {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }
    std::string json = serializeJson(result);
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        if (!running_.load(std::memory_order_relaxed)) {
            return;
        }
        outboundQueue_.push(std::move(json));
    }
    queueCv_.notify_one();
}

void WebhookResultWriter::close() {
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        if (!senderThread_.joinable()) {
            curl_.reset();
            return;
        }
        running_.store(false, std::memory_order_release);
    }
    queueCv_.notify_all();
    if (senderThread_.joinable()) {
        senderThread_.join();
    }
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        while (!outboundQueue_.empty()) {
            outboundQueue_.pop();
        }
        curl_.reset();
    }
    curlGlobalReleaseRef();
}

std::string WebhookResultWriter::serializeJson(const FrameResult& result) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(7);
    oss << "{\"session_id\":\"" << escapeJsonString(sessionId_) << "\",\"frame_id\":"
        << result.frame_id << ",\"timestamp_utc\":\"" << escapeJsonString(result.timestamp_utc)
        << "\",\"source_id\":\"" << escapeJsonString(result.source_id) << "\",";
    appendDetectionsArray(oss, result);
    oss << '}';
    return oss.str();
}

size_t WebhookResultWriter::discardWriteCallback(char* ptr, size_t size, size_t nmemb,
                                                 void* userdata) {
    (void)ptr;
    (void)userdata;
    return size * nmemb;
}

bool WebhookResultWriter::postJsonOnce(const std::string& json) const {
    CURL* handle = curl_.get();
    if (!handle) {
        return false;
    }
    curl_easy_setopt(handle, CURLOPT_URL, callbackUrl_.c_str());
    curl_easy_setopt(handle, CURLOPT_POST, 1L);
    curl_easy_setopt(handle, CURLOPT_POSTFIELDS, json.data());
    curl_easy_setopt(handle, CURLOPT_POSTFIELDSIZE, static_cast<long>(json.size()));
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, &WebhookResultWriter::discardWriteCallback);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, nullptr);
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(handle, CURLOPT_HTTPHEADER, headers);
    const CURLcode code = curl_easy_perform(handle);
    long httpCode = 0;
    if (code == CURLE_OK) {
        curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &httpCode);
    }
    curl_slist_free_all(headers);
    return code == CURLE_OK && httpCode >= 200 && httpCode < 300;
}

void WebhookResultWriter::postJsonWithRetries(const std::string& json) {
    if (postJsonOnce(json)) {
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (postJsonOnce(json)) {
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    if (postJsonOnce(json)) {
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    if (postJsonOnce(json)) {
        return;
    }
    fprintf(stderr,
            "WebhookResultWriter: dropped message after failed POST retries (url=%s)\n",
            callbackUrl_.c_str());
}

void WebhookResultWriter::senderLoop() {
    for (;;) {
        std::unique_lock<std::mutex> lock(queueMutex_);
        queueCv_.wait(lock, [this] {
            return !outboundQueue_.empty() || !running_.load(std::memory_order_acquire);
        });
        while (!outboundQueue_.empty()) {
            std::string json = std::move(outboundQueue_.front());
            outboundQueue_.pop();
            lock.unlock();
            postJsonWithRetries(json);
            lock.lock();
        }
        if (!running_.load(std::memory_order_acquire)) {
            break;
        }
    }
}
