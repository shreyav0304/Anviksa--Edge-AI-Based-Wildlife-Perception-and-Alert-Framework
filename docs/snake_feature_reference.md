# Snake Feature Reference — Venomous vs Non-Venomous Snakes

This reference is for documentation and explanation only. It is not an identification key, a rule-based classifier, or a replacement for the trained Anviksa MobileNetV3 Large model.

## Feature comparison

| Property | Venomous | Non-Venomous |
|---|---|---|
| Venom | Produce venom used for prey capture or defense. | Do not use a venom-injection system for envenomation. |
| Fangs | Usually possess specialized venom-delivery fangs. | Generally lack specialized venom-injecting fangs. |
| Bite | A bite may inject venom and can cause serious medical effects. | Usually causes mechanical injury such as puncture wounds. |
| Head shape | Some species may have broad or distinctive heads. **Head shape is not a reliable standalone venom indicator.** | Head shape also varies significantly. **Head shape is not a reliable standalone venom indicator.** |
| Pupil shape | Can have vertical or round pupils depending on species. **Pupil shape is not a reliable standalone indicator.** | Can also have different pupil shapes. **Pupil shape is not a reliable standalone indicator.** |
| Body structure | May be stout or slender depending on species. **Body shape alone must not be used for venom classification.** | May also be stout or slender. **Body shape alone must not be used for venom classification.** |
| Defense | May bite, display warning behavior, or use venom. | May bite, constrict, flee, or use other defensive behaviors. |
| Prey capture | Often immobilize prey using venom. | May constrict prey or swallow prey without venom injection. |
| Examples in India | Indian Cobra; Common Krait; Russell's Viper; Saw-scaled Viper. | Indian Rock Python; Rat Snake; Indian Trinket Snake. |
| Human risk | Some species may cause severe or life-threatening envenomation. | Generally lower medical risk, although bites can still cause injury. |

## Critical identification note

> **Venomous and non-venomous snakes cannot be reliably distinguished using a single visible feature such as head shape, pupil shape, color, or body structure. These characteristics contain many species-level exceptions and must not be treated as deterministic identification rules.**

**Anviksa does not use manually programmed rules such as triangular head = venomous or round pupil = non-venomous.**

## Why Deep Learning Is Used in Anviksa

Snake appearance varies significantly across species, and venomous and non-venomous snakes may have visually similar patterns. Simple handcrafted rules are therefore unreliable. Anviksa uses CNN-based deep learning to learn visual features from labeled training images, with MobileNetV3 Large selected as the classification model. In the intended application layer, OpenCV handles camera-frame and image preprocessing while preserving the validated model input contract; live camera-stream integration remains planned. The final model predicts one of its trained classes rather than applying manually written snake-shape rules.

The visual similarity between venomous and non-venomous snakes makes reliable classification challenging because characteristics such as head shape, body structure, pupil shape, and coloration are not universally reliable indicators. Therefore, Anviksa uses a trained CNN-based deep learning model to learn discriminative visual features from labeled snake images rather than depending on manually defined physical rules.

## External image test documentation

The current seven-image external test is a qualitative inference test. Its recorded results are:

- 7 external snake images
- 3 predicted `Venomous_Snake`
- 4 predicted `Non_Venomous_Snake`
- 6 high-confidence predictions
- 1 moderate-confidence prediction

Ground-truth venom status is currently unavailable. Consequently:

- Do not calculate external accuracy.
- Do not mark model predictions as correct or incorrect.
- Do not use visible snake features to invent ground-truth labels.

## Safety note

> **Model predictions are experimental AI outputs and must not be treated as authoritative snake-safety identification. Unknown snakes should not be approached or handled based on the model prediction.**

