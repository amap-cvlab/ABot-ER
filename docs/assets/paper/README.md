# Figures from the technical report / 论文插图

Paper: [ABot-ER on arXiv](https://arxiv.org/abs/2610.18355).
The overview and architecture assets are raster renders of figure regions from the report revision.
No plots, labels, example outputs, or numerical values were redrawn.
Only the surrounding page text and margins were excluded.
图片直接取自用户提供的论文，仅去除周围正文与页边距，不重绘内容或修改数值。

Figure-source revision SHA-256 (not a checksum of the current arXiv download):
`11249571be1cbafee54c2227dca3c5a6509bdc9083ac02b24a4f4f0ff4ccf073`

| Asset | Report figure | PDF page (1-based) | Crop rectangle (PDF points, top-left origin) |
| --- | ---: | ---: | --- |
| [Benchmark overview](benchmark-overview.png) | 1 | 1 | 71, 414, 541, 643 |
| [Capability overview](capability-overview.png) | 2 | 5 | 70, 80, 543, 343 |
| [Model architecture](model-architecture.png) | 3 | 8 | 72, 160, 540, 436 |

The [real-world demonstrations](real-world-demonstrations.jpg) use the
author-supplied privacy-redacted figure (1632 × 1056 pixels), not the original
unredacted PDF crop. Its SHA-256 is
`3718ccc7dcaefadc903c1bcd2b6149994b3bc48663e786dd28449db452c2656b`.
真机演示使用作者提供的打码版本，不提供未打码原图。

Rendering: PyMuPDF, scale 4 pixels per PDF point (288 dpi), RGB with a white
background. The README displays each image at its container width and links to
the full-resolution asset. Both language versions use the same images.

The benchmark comparison is a report snapshot, not a live leaderboard.
The capability-example cards come from evaluation splits, not training data.
The architecture diagram describes the 9B configuration; EQA is a task label,
not a replacement for the implementation's `[2D_QA]` token.
The physical demonstrations are qualitative examples, not benchmark scores.
Source datasets and third-party materials retain their applicable rights.
