#pragma once

#include "output/result_writer.h"
#include <vector>
#include <memory>

/// @brief Dispatches FrameResult to all registered ResultWriters.
///
/// Add writers via addWriter() during pipeline init. The handler does
/// not own the output format logic -- that lives in each writer impl.
/// To add a new output, create a ResultWriter subclass and register it.
class ResultHandler {
public:
    ResultHandler() = default;
    ~ResultHandler();

    ResultHandler(const ResultHandler&) = delete;
    ResultHandler& operator=(const ResultHandler&) = delete;

    /// @brief Register a writer. Ownership is transferred.
    void addWriter(std::unique_ptr<ResultWriter> writer);

    /// @brief Open all registered writers. Call once before handle().
    bool openAll();

    /// @brief Dispatch result to every registered writer.
    void handle(const FrameResult& result);

    /// @brief Flush and close all writers (called during shutdown).
    void closeAll();

private:
    std::vector<std::unique_ptr<ResultWriter>> writers_;
};
