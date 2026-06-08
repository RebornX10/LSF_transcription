# Trained model hook (optional)

If you train a sequence classifier and drop it here, the recogniser uses it
automatically (in preference to the DTW templates).

Expected files:

| File                  | Purpose                                            |
|-----------------------|----------------------------------------------------|
| `model.h5`            | Keras model saved with `model.save("model.h5")`    |
| `model.labels.json`   | JSON list of class names, index-aligned to outputs |

**Input shape:** `(batch, 32, 153)` — i.e. each sample is a sign segment
resampled to `TEMPLATE_LEN = 32` frames of the *matching vector* (pose block +
both hands), `9*3 + 21*3*2 = 153` features per frame. This is exactly the array
the recogniser passes to `model.predict(...)` (see
`recognizer.SignRecognizer._classify_model`).

**Output:** a softmax over the classes listed in `model.labels.json`. Anything
below 0.5 confidence is ignored.

To build a dataset, record signs through the UI (they land in
`signs/lsf_signs.json` as `(32, 153)` sequences per gloss) and use those as
labelled training samples, augmenting with time/scale jitter.
