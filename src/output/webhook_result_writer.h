#pragma once

#include "output/result_writer.h"
#include <curl/curl.h>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

/// @brief Asynchronous HTTP POST writer that delivers frame analytics to a webhook URL.
///
/// Serializes each @ref FrameResult to JSON and POSTs it to @p callback_url with
/// Content-Type application/json. @c write() is non-blocking; a background thread
/// drains an internal queue and performs the HTTP requests with retry and backoff.
class WebhookResultWriter : public ResultWriter {
public:
    /// @param callback_url Destination URL for HTTP POST callbacks.
    /// @param session_id   Session identifier included in every JSON payload.
    explicit WebhookResultWriter(std::string callback_url, std::string session_id);
    ~WebhookResultWriter() override;
    WebhookResultWriter(const WebhookResultWriter&) = delete;
    WebhookResultWriter& operator=(const WebhookResultWriter&) = delete;
    /// @brief Initializes libcurl (global, ref-counted) and starts the sender thread.
    /// @return true if setup succeeded.
    bool open() override;
    /// @brief Enqueues a JSON payload for the sender thread; returns immediately.
    void write(const FrameResult& result) override;
    /// @brief Stops the sender thread after draining the queue and releases the curl handle.
    void close() override;

private:
    /// @brief Builds the JSON body for one frame (manual serialization, no external JSON library).
    std::string serializeJson(const FrameResult& result) const;
    /// @brief Background loop: dequeue JSON strings and POST with retries.
    void senderLoop();
    /// @brief Performs a single HTTP POST; returns true on transport and 2xx HTTP success.
    bool postJsonOnce(const std::string& json) const;
    /// @brief POSTs with up to four attempts and 100ms / 500ms / 1000ms delays between failures.
    void postJsonWithRetries(const std::string& json);
    static size_t discardWriteCallback(char* ptr, size_t size, size_t nmemb, void* userdata);
    std::string callbackUrl_;
    std::string sessionId_;
    std::mutex queueMutex_;
    std::condition_variable queueCv_;
    std::queue<std::string> outboundQueue_;
    std::atomic<bool> running_{false};
    std::thread senderThread_;
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl_{nullptr, curl_easy_cleanup};
};
