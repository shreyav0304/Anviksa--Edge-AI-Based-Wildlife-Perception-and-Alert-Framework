# Repository checkup - 12 October 2026

## Scope and results

All 90 Python sources in the working repository parse successfully; all 64
tracked JSON artifacts parse successfully. The deployed Float32, Float16 and
Wildlife Gate model files match their recorded SHA-256 hashes. Training data,
model binaries, and validated experiment results were preserved.

The dashboard data checks pass from the published summaries and prediction
caches, without private training runs. The completed confusion matrices have
545 predictions each: Custom CNN 275 correct, MobileNetV2 504 correct, and
MobileNetV3 Large 520 correct. EfficientNetB0 remains incomplete with N/A results.

## Fixes

- Dashboard pages and package exports load lazily. Missing OpenCV or a plotting
  dependency no longer blocks startup or unrelated sections. Failed sections
  show their error, setup instructions, and a Retry button.
- All eight navigation entries remain reachable on small displays; desktop
  regression checks cover 800x480 and 480x320. Native desktop fonts, label/card
  styles, scrolling over navigation buttons, and measured content widths
  improve visibility. Repeated page retries do not accumulate wheel handlers.
- Page refresh and callback failures appear in terminal diagnostics and the
  status bar instead of being swallowed. Launch errors explain missing
  dependencies or a missing graphical desktop session.
- Video history events use a separate bounded delivery queue so dropping old
  preview frames cannot discard eligible detections. Snapshots use the exact
  analyzed frame and immutable session source identity; pending eligible events
  are drained on stop. Frame timestamps use zero-based video timing.
- Only the capture thread releases OpenCV's native camera object. The UI waits
  for a stopping thread before allowing a replacement session. Capture
  constructor failures become error updates rather than leaving the UI stuck.
- Invalid frame-skip settings, nonfinite/out-of-range pixels and probabilities,
  reordered class labels, mismatched tensor interfaces, and a changed Wildlife
  Gate binary fail explicitly before producing misleading results.
- Late high-confidence stable sightings are counted correctly. Re-analysis
  failure clears the previous saveable image result; closing cancels pending
  image polling callbacks.
- Truncated, nonfinite and invalid-boolean history rows are skipped and reported.
  Legacy history without optional video fields remains readable and appendable.
  Filtered export cannot overwrite the master history. The history table has
  both scrollbars, and expired cooldown keys are pruned.
- Webcam diagnostics render through Tkinter, matching the headless OpenCV
  dependency. Package verification reports malformed JSON as failed checks.
  Benchmarks explain missing input folders, record input hashes, and reject
  comparisons with different inputs, devices or measurement protocols.
- README and Pi setup instructions describe the implemented runtime features
  and provide reproducible health and regression commands.

## Validation performed

- 27 regression checks pass on Windows, including five off-screen desktop tests.
  Camera behavior is simulated in these focused regression tests.
- The same suite passes under Linux for its 22 non-GUI checks; five desktop
  checks are skipped because that run has no graphical display.
- Real Linux CPU inference succeeds for both TFLite representations on three
  existing external images; the predicted classes agree across representations.
  Wildlife Gate allocation and a synthetic-image forward pass succeed.
- Real OpenCV decoding and the complete gate/species video worker succeed on a
  temporary six-frame synthetic MJPG clip, with frame skipping, three processed
  frames, normal end-of-file handling and capture release. OpenCV was installed
  in an isolated test cache; the existing Linux environment was not modified.
- Read-only package and dashboard health checks pass, and Git whitespace checks
  pass.

Run the portable checks from the activated app environment:

    python -B pi_deployment/verify_package.py
    python -B scripts/check_dashboard.py
    python -B -m unittest discover -s tests -v

Include desktop checks from an active graphical login:

    ANVIKSA_GUI_TESTS=1 python -B -m unittest discover -s tests -v

## Remaining external validation

No physical Raspberry Pi was connected to this session. Raspberry Pi ARM64
installation, its display session, USB camera drivers/codecs, thermal behavior,
memory and FPS measurements require on-device validation. Linux CPU checks used
an existing x86 development environment and TensorFlow's interpreter fallback;
they do not establish Pi performance or ARM64 wheel compatibility.

The local Windows development environment lacks OpenCV and a TFLite interpreter;
desktop tests used its compatible NumPy 1.x environment and did not run Windows
inference. Follow the documented Pi dependency profile on the target device.

mmWave triggers, GPIO LED/buzzer integration and solar field hardware remain
planned. Training and conversion were not rerun, and private dataset utilities
that require excluded data cannot be validated end-to-end from this checkout.
A native camera read that blocks indefinitely still needs driver/device recovery;
the application avoids releasing the capture concurrently from another thread.
