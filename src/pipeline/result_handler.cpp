#include "pipeline/result_handler.h"

ResultHandler::~ResultHandler() {
    closeAll();
}

void ResultHandler::addWriter(std::unique_ptr<ResultWriter> writer) {
    writers_.push_back(std::move(writer));
}

bool ResultHandler::openAll() {
    for (auto& w : writers_) {
        if (!w->open()) return false;
    }
    return true;
}

void ResultHandler::handle(const FrameResult& result) {
    for (auto& w : writers_) {
        w->write(result);
    }
}

void ResultHandler::closeAll() {
    for (auto& w : writers_) {
        w->close();
    }
}
