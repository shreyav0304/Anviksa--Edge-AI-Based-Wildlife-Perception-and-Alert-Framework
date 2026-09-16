# Proposed MobileNetV3 Large experiment — design only

This candidate augments the existing seven-class wildlife classifier. It does not define a snake-only production model and does not replace MobileNetV3 Large V1.

Build a new, versioned experimental training input from the existing clean training and validation data. Add only the candidate's `TRAIN` snake frames to the corresponding `Venomous_Snake` and `Non_Venomous_Snake` training classes. Add only its `VALIDATION` snake frames to those validation classes. Continue using the existing clean data for Cow, Deer, Elephant, Monkey, and Wild_Boar, as well as the existing clean snake data. Preserve the seven-class output order exactly:

1. Cow
2. Deer
3. Elephant
4. Monkey
5. Non_Venomous_Snake
6. Venomous_Snake
7. Wild_Boar

Train a new checkpoint and retain MobileNetV3 Large V1 unchanged. Use validation only for epoch selection, hyperparameters, model selection, calibration, and thresholds. After the experiment is fixed, evaluate once on the existing frozen image test and separately on this candidate's `INDEPENDENT_VIDEO_TEST`. Never copy independent video-test frames into training or validation, and never merge the video test with the frozen image benchmark.

Each approved video frame is an individual image sample during MobileNetV3 training. These frames expose the classifier to different snake positions, body orientations, movement states, partial visibility, backgrounds, lighting, distances, and camouflage conditions. MobileNetV3 still performs per-frame image classification and does not learn temporal relationships between frames. Temporal sequence learning is a separate future experiment.

Monitor per-class recall and confusion for all seven classes, not only aggregate accuracy. Account for correlated frames from one source video when batching or weighting so high-frame-count videos do not dominate. Keep the manifest's video and split-group boundaries in all loading, balancing, evaluation, and error analysis.

Only four verified nonvenomous source videos exist. Two are in training, one in validation, and one in independent video test. This provides class presence in every split but weak estimates of nonvenomous generalization: validation and video-test results each depend on one nonvenomous video, and all four nonvenomous videos are web sourced. Treat their metrics as preliminary and collect additional verified rescuer/nonweb nonvenomous videos before drawing strong conclusions.
