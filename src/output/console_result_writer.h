#pragma once

#include "output/result_writer.h"

/// @brief Prints per-frame detection results to stdout (the original behavior).
class ConsoleResultWriter : public ResultWriter {
public:
    /// @param verbose If true, print per-detection bbox details. If false, only summary.
    explicit ConsoleResultWriter(bool verbose = true) : verbose_(verbose) {}

    void write(const FrameResult& result) override;

private:
    bool verbose_;
};
