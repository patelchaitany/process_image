#include "output/console_result_writer.h"
#include "postprocess/nms.h"
#include "matching/face_matcher.h"
#include <cstdio>

void ConsoleResultWriter::write(const FrameResult& result) {
    printf("Frame %lu: %zu faces detected\n",
           result.frame_id, result.detections.size());

    if (!verbose_) return;

    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& det = result.detections[i];
        const char* name = "unknown";
        float conf = 0.0f;

        if (i < result.matches.size() && result.matches[i].faceId >= 0) {
            name = result.matches[i].name.c_str();
            conf = result.matches[i].confidence;
        }

        printf("  [%zu] bbox=(%.0f,%.0f,%.0f,%.0f) det_conf=%.3f match=%s (%.3f)\n",
               i, det.x1, det.y1, det.x2, det.y2, det.confidence, name, conf);
    }
}
