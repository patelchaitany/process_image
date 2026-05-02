#pragma once

#include "postprocess/nms.h"
#include "matching/face_matcher.h"
#include <vector>
#include <cstdint>

struct FrameResult {
    uint64_t frame_id;
    std::vector<Detection> detections;
    std::vector<MatchResult> matches;
};

class ResultHandler {
public:
    void handle(const FrameResult& result);
};
