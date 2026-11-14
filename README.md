# ELMA

ELMA combines spatial–audiovisual conditioning with speaker memory and recurrent history for consistent target-speech extraction from six-channel mixtures. It processes 1 s windows and emits 0.4 s per step.

Install the dependencies and prepare local model/data assets as described in [docs/setup.md](docs/setup.md). Set `$SAVE` to the dataset directory and `$OUTPUT` to an output directory with sufficient storage.

Fine-tune visually conditioned IPDNet2, DAVE, the target selector and speaker identity projection on all 32,768 training scenes:

```bash
python -m hisav train --data "$SAVE" --bundle assets/default --output "$OUTPUT/training"
```

Infer on all 2,000 test scenes:

```bash
python -m hisav infer --data "$SAVE" --bundle assets/default --output "$OUTPUT/evaluation"
```

See [docs/recipe.md](docs/recipe.md) for the training recipe. Model weights and dataset files are external assets and are not included in this code-only review snapshot. [Third-party notices](THIRD_PARTY_NOTICES.md) are retained.
