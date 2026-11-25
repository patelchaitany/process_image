# fastpipe-dev

Everything around the pipeline that is not the pipeline: measurements, evaluators and test scripts. The pipeline
itself is in [`../fastpipe/`](../fastpipe/README.md) and does not depend on anything here.

| Path | Content |
|---|---|
| `benchmarks/models/RESULTS.md` | Model comparison and the recogniser choice: accuracy, size, speed, thresholds, licences |
| `benchmarks/models/CANDIDATES.md` | Every model considered, with source and licence |
| `benchmarks/models/results/` | Raw result files behind the tables |
| `benchmarks/models/eval/` | The evaluators, with the batch files that ran them in `eval/run/` |
| `tools/bench/` | Stress runs and profiling |
| `tools/check/` | Checks against a reference: embeddings, boxes, tracker, match scores |
| `tools/clips/` | Make the test clips with FFmpeg |
| `tools/int8/` | The 8-bit recogniser experiment |
| `tools/live/` | Test the camera page without a camera |
| `tools/exp/` | Run one experiment on the remote GPU machine under a lock |

Every script is listed with its purpose in [tools/README.md](tools/README.md).
