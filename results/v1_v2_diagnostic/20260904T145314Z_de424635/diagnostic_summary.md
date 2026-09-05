# ANVIKSA V1 vs V2 Diagnostic Summary

## A. Executive Summary

V2 changed 14 of 545 predictions. It fixed 4 V1 errors but broke 10 V1-correct cases, for a net correction of -6. Venomous recall improved, while Non-Venomous recall declined. V1 remains the selected model.

## B. Integrity Verification

All required artifacts exist. Dataset V2 fingerprint is `9dd48f6a1aeb8afad1f76d943b7baf36a58c837e50b7124cacb3f5211e580f23`. Test count and order match at 545. Cached predictions were used; no inference, training, or fine-tuning occurred. Protected paths remained unchanged: True.

## C. Prediction Transition Analysis

Both correct: 510; V1 correct to V2 wrong: 10; V1 wrong to V2 correct: 4; both wrong: 21. Net correction: -6.

## D. Snake-Specific Error Analysis

Non-Venomous correct to Venomous: 8. Venomous errors corrected: 4. Largest worsened off-diagonal cells: [(8, 'Non_Venomous_Snake', 'Venomous_Snake'), (1, 'Venomous_Snake', 'Deer')]. Largest improved cells: [(-3, 'Venomous_Snake', 'Non_Venomous_Snake')].

## E. Probability / Confidence Shift

Mean venomous probability shift across actual snakes: 0.043608586; median: 0.000709414. Counts toward Venomous / toward Non-Venomous / unchanged: 179 / 75 / 15 (tolerance 1e-06). New V2 snake errors were chiefly low confidence; uncertainty count 5.

## F. New Dataset Species Distribution

Dominant Venomous species: Common Krait (21). Dominant Non-Venomous species: Checkered Keelback (21). Species counts are metadata-derived only. DOMAIN SHIFT: NOT PROVEN.

## G. Dataset Balance

V1 train Non-Venomous/Venomous: 598/889 (ratio 0.672666). V2: 663/932 (ratio 0.711373). Balance moved numerically closer to parity, but that alone does not explain behavior.

## H. Training History

Stage 1 best: epoch 13, validation accuracy 0.940559447. Stage 2 best: epoch 7, 0.933566451. Stage 2 never exceeded Stage 1. Its validation loss did improve transiently while training accuracy increased; later divergence is consistent with possible overfitting, not proof of it.

## I. V1/V2 Configuration Comparability

Comparability is HIGH. Architecture, fresh ImageNet initialization strategy, preprocessing, augmentation, batch size, optimizer, learning rates, fine-tune depth, BatchNorm policy, callbacks, seed, and epoch limits match. Dataset contents, derived class weights, and selected training stage differ; therefore observed effects cannot be attributed to image content alone independently of frequency-derived weights and stochastic training.

## J. Class Weight Analysis

Weights changed. Non-Venomous: 0.886526517 to 0.822882999; Venomous: 0.596336172 to 0.585377069. Both declined, with a larger relative decline for Non-Venomous.

## K. Evidence-Supported Findings

The clearest measured mechanism is a decision-boundary/probability shift toward Venomous on actual Non-Venomous test images, coinciding with changed snake class frequencies and class weights. This explains the prediction pattern but does not prove which property of the added species/source domains caused it; domain shift is not proven.

## L. Unproven Hypotheses

Specific species concentration, source-domain characteristics, visual label noise, or augmentation interactions may contribute, but the stored metadata and frozen-test transitions do not establish causality. DOMAIN SHIFT: NOT PROVEN.

## M. Recommended Next Experiment

**SUPPORTED BY DIAGNOSTIC EVIDENCE:** retain V1 for deployment and inspect the identified transition/contact-sheet groups plus rights status before any further training.

**HYPOTHESIS REQUIRING EXPERIMENT:** a future separately approved controlled ablation could isolate added-image content from recalculated class weights. No V3 experiment was started here.

V2 deployment promotion: **NOT APPROVED**. Dataset image rights remain **REVIEW REQUIRED**.
