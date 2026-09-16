# MobileNetV3 Large video-frame v1 — controlled experiment

## Configuration

This single experimental seven-class run reproduced the official V1 method: MobileNetV3 Large with ImageNet initialization, 224 × 224 RGB input, Keras internal MobileNetV3 preprocessing to `[-1, 1]`, global average pooling, Dense(256, ReLU), Dropout(0.35), and seven-way softmax in the authoritative order `Cow, Deer, Elephant, Monkey, Non_Venomous_Snake, Venomous_Snake, Wild_Boar`. Augmentation used horizontal flip, rotation 0.055, height/width translation 0.10, zoom 0.15, contrast 0.10, and brightness 0.10.

Training used seed 42, batch size 32, balanced class weights, Adam and categorical cross-entropy. Stage 1 froze the ImageNet backbone with learning rate 0.001 and a maximum of 30 epochs. Fine-tuning exposed the upper 45 backbone layers (36 trainable after freezing BatchNormalization), used learning rate 0.00001, and allowed 25 epochs. Both stages used `ModelCheckpoint(val_accuracy, max)`, `EarlyStopping(val_loss, patience=6, restore_best_weights=True)`, and `ReduceLROnPlateau(val_loss, factor=0.5, patience=3, min_lr=1e-7)`. The runtime software and CPU-only environment exactly match the saved V1 environment versions. No V1 setting needed reconstruction beyond its saved artifacts.

## Data

Clean train counts were Cow 316, Deer 579, Elephant 638, Monkey 227, Non_Venomous_Snake 598, Venomous_Snake 889, Wild_Boar 464. Added video-train counts were 51 nonvenomous and 152 venomous, producing 3,914 combined training samples with snake totals 649 and 1,041.

Clean validation counts were Cow 40, Deer 74, Elephant 81, Monkey 29, Non_Venomous_Snake 104, Venomous_Snake 158, Wild_Boar 59. Added video-validation counts were 11 nonvenomous and 18 venomous, producing 574 combined validation samples with snake totals 115 and 176. All preflight leakage and eligibility checks passed. The 38-frame independent video test stayed sealed during training and checkpoint selection.

## Training and checkpoint selection

Stage 1 completed 18 epochs; its best validation accuracy was 0.930314 at stage epoch 18. Fine-tuning completed 15 epochs; its best validation accuracy was 0.932056 at stage epoch 7 (global epoch 25). Selection used maximum combined-validation accuracy across stage checkpoints. The selected model is `selected_model.keras`, sourced from `fine_tuned_best.weights.h5`, SHA-256 `4f688ea8eb85f584efa89a1095a8b4ea685ec17c0714d9f0a24de0b5d0fd708f`. Its hash was recorded at `2026-09-16T06:01:14.808343+00:00`, before the sealed test was opened at `2026-09-16T06:01:42.153375+00:00`.

## Combined validation (development)

Accuracy 0.932056; macro precision 0.955518; macro recall 0.960616; macro F1 0.957543. Per-class precision/recall/F1: Cow 0.975610/1.000000/0.987654; Deer 1/1/1; Elephant 0.975610/0.987654/0.981595; Monkey 1/1/1; Non_Venomous_Snake 0.811024/0.895652/0.851240; Venomous_Snake 0.926380/0.857955/0.890855; Wild_Boar 1.000000/0.983051/0.991453.

## Video-validation subset (development)

Accuracy 0.689655; two-class macro precision 0.775000; macro recall 0.750000; macro F1 0.688172. Non_Venomous_Snake precision/recall/F1 was 0.550000/1.000000/0.709677. Venomous_Snake was 1.000000/0.500000/0.666667. Of 11 nonvenomous frames, all 11 were correct. Of 18 venomous frames, 9 were correct and 9 were predicted nonvenomous.

## Independent video test

The sealed test was evaluated once after checkpoint hashing. Accuracy was 0.736842; two-class macro precision 0.761905; macro recall 0.814815; macro F1 0.730114. Non_Venomous_Snake precision/recall/F1 was 1.000000/0.629630/0.772727. Venomous_Snake was 0.523810/1.000000/0.687500.

`VID_2fa5a241b315c1a7` (Bitis nasicornis, venomous): 11/11 correct, 0 incorrect, accuracy 1.0, all 11 predictions Venomous_Snake. `VID_babc3e7bef3c3d3e` (Python regius, nonvenomous): 17/27 correct, 10 incorrect, accuracy 0.629630, with 17 Non_Venomous_Snake and 10 Venomous_Snake predictions.

## Frozen image test and V1 comparison

The experimental checkpoint achieved accuracy 0.946789, macro precision 0.963438, macro recall 0.964747, and macro F1 0.964019. Snake macro recall was 0.907746 and snake macro F1 was 0.910602. Non_Venomous_Snake precision/recall/F1 was 0.907692/0.921875/0.914729; Venomous_Snake was 0.919708/0.893617/0.906475. Snake cross-errors were NV→V 10 and V→NV 12.

Against official V1, accuracy changed from 0.954128 to 0.946789 (-0.734 percentage points); macro precision 0.968867 to 0.963438 (-0.543 pp); macro recall 0.968697 to 0.964747 (-0.395 pp); macro F1 0.968725 to 0.964019 (-0.471 pp); snake macro recall 0.921570 to 0.907746 (-1.382 pp); and snake macro F1 0.923718 to 0.910602 (-1.312 pp). Nonvenomous recall increased from 0.914063 to 0.921875 (+0.781 pp), while venomous recall decreased from 0.929078 to 0.893617 (-3.546 pp). NV→V errors decreased from 11 to 10; V→NV errors increased from 8 to 12. The experiment is not promoted.

## Interpretation and limitations

Each approved frame was treated as an independent image sample. MobileNetV3 remains a per-frame classifier and learned no temporal relationships. The independent test represents only two source videos—one venomous snake and one nonvenomous snake—rather than 38 independent snakes, so its result is preliminary. Only four verified nonvenomous videos existed, all web sourced. The development video subset also showed substantial venomous-to-nonvenomous confusion. These constraints preclude broad claims about video generalization or production superiority.
