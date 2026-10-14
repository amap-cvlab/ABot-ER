# Output-contract checklist

This is a checklist based on the agreed project protocol. Import the actual frozen evaluation templates and code hashes before marking any contract verified.

| Task | Expected output |
| --- | --- |
| VABench-P | `<Answer><point>[[x1,y1],...,[x8,y8]]</point></Answer>`; exactly 8 points, original-image-normalized [0,1] |
| Where2Place-point | `[(x1, y1), ...]`; [0,1]; no added exactly-N instruction |
| EmbSpatial / ERQA / Cosmos | Single A–D choice |
| VSI | Numeric or MCQ according to the actual subtask parser |
| OpenEQA / RoboVQA | Native concise text |
| UniEQA | Text or option according to task schema |
| ET-Plan / EB-ALFRED / EB-Habitat | Independently frozen planning schemas and action spaces |

Training point serialization uses three decimals. Retain the scorer's documented scaling and rounding rules; do not silently convert model mistakes. VABench scoring and point uniqueness policy must be recorded independently. W2P output-point count policy must likewise match the frozen scorer.

Video contracts include exact frame sampling, ordering and short-video behavior. Judge failures are reported separately from valid zero scores. Route injection is model-conditioned and verified after tokenization.
