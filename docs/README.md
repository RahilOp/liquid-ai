# Cross-cutting docs

- `asr-ft-training-log.md` — full ASR LoRA training log: dataset taxonomy, MUSAN augmentation,
  LoRA config evolution, all 7 training runs with val-loss curves, checkpoint table, next steps.
- `eval-ft-context.md` / `eval-ft-readme.md` — evaluation methodology, metric definitions,
  frozen-benchmark design, baseline results, and the forgetting analysis.
- `kaigi-system-architecture.md` — Kaigi assistant architecture (base + 3 adapters, cascade, latency).
- `kaigi-running.md` — how to run the assistant (GPU + CPU paths).

More detailed component docs (transcription convention, translation cascade, minutes, dataset
research) live in `../05-app-and-demo/kaigi-app/docs/` and `../05-app-and-demo/kaigi-app/research/`.
