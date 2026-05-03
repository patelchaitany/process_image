#include "prefetch_source.h"
#include <cstdio>

PrefetchSource::PrefetchSource(std::unique_ptr<FrameSource> inner, int bufferSize)
    : inner_(std::move(inner)), bufferSize_(bufferSize) {}

PrefetchSource::~PrefetchSource() {
    close();
}

bool PrefetchSource::open(const std::string& source) {
    if (!inner_->open(source)) return false;

    running_ = true;
    eof_ = false;
    decodeThread_ = std::thread(&PrefetchSource::decodeLoop, this);

    fprintf(stderr, "PrefetchSource: started decode thread (buffer=%d frames)\n", bufferSize_);
    return true;
}

bool PrefetchSource::read(Frame& frame) {
    std::unique_lock<std::mutex> lock(mu_);

    notEmpty_.wait(lock, [this] {
        return !queue_.empty() || eof_.load(std::memory_order_relaxed);
    });

    if (queue_.empty()) return false;

    frame = std::move(queue_.front());
    queue_.pop();

    lock.unlock();
    notFull_.notify_one();
    return true;
}

void PrefetchSource::decodeLoop() {
    while (running_.load(std::memory_order_relaxed)) {
        Frame frame;
        bool ok = inner_->read(frame);

        if (!ok) {
            eof_ = true;
            notEmpty_.notify_one();
            return;
        }

        std::unique_lock<std::mutex> lock(mu_);
        notFull_.wait(lock, [this] {
            return static_cast<int>(queue_.size()) < bufferSize_ ||
                   !running_.load(std::memory_order_relaxed);
        });

        if (!running_.load(std::memory_order_relaxed)) return;

        queue_.push(std::move(frame));
        lock.unlock();
        notEmpty_.notify_one();
    }
}

void PrefetchSource::close() {
    running_ = false;
    notFull_.notify_all();
    notEmpty_.notify_all();

    if (decodeThread_.joinable()) {
        decodeThread_.join();
    }

    {
        std::lock_guard<std::mutex> lock(mu_);
        while (!queue_.empty()) queue_.pop();
    }

    inner_->close();
}

int PrefetchSource::width() const { return inner_->width(); }
int PrefetchSource::height() const { return inner_->height(); }
double PrefetchSource::fps() const { return inner_->fps(); }
bool PrefetchSource::is_open() const { return inner_->is_open(); }
