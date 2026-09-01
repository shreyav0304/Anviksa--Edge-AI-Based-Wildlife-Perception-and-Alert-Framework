"""Descriptive snake-feature metadata for documentation and GUI explanations.

This module intentionally contains no identification or classification logic.
"""

SNAKE_FEATURE_REFERENCE = {
    "venomous": {
        "venom": "Produce venom used for prey capture or defense.",
        "fangs": "Usually possess specialized venom-delivery fangs.",
        "bite": "A bite may inject venom and can cause serious medical effects.",
        "head_shape": "Some species may have broad or distinctive heads; not reliable as a standalone venom indicator.",
        "pupil_shape": "May have vertical or round pupils depending on species; not reliable as a standalone indicator.",
        "body_structure": "May be stout or slender depending on species; body shape alone must not be used for venom classification.",
        "defense": "May bite, display warning behavior, or use venom.",
        "prey_capture": "Often immobilize prey using venom.",
        "examples_in_india": (
            "Indian Cobra",
            "Common Krait",
            "Russell's Viper",
            "Saw-scaled Viper",
        ),
        "human_risk": "Some species may cause severe or life-threatening envenomation.",
    },
    "non_venomous": {
        "venom": "Do not use a venom-injection system for envenomation.",
        "fangs": "Generally lack specialized venom-injecting fangs.",
        "bite": "Usually causes mechanical injury such as puncture wounds.",
        "head_shape": "Head shape varies significantly; not reliable as a standalone venom indicator.",
        "pupil_shape": "May have different pupil shapes; not reliable as a standalone indicator.",
        "body_structure": "May be stout or slender; body shape alone must not be used for venom classification.",
        "defense": "May bite, constrict, flee, or use other defensive behaviors.",
        "prey_capture": "May constrict prey or swallow prey without venom injection.",
        "examples_in_india": (
            "Indian Rock Python",
            "Rat Snake",
            "Indian Trinket Snake",
        ),
        "human_risk": "Generally lower medical risk, although bites can still cause injury.",
    },
}

IDENTIFICATION_WARNING = (
    "Head shape, pupil shape, body shape, and color must not be used "
    "as deterministic rules for venomous/non-venomous classification."
)

MODEL_CONTEXT = {
    "purpose": "Documentation and GUI explanation only.",
    "selected_model": "MobileNetV3 Large",
    "classification_approach": "CNN-based deep learning from labeled images.",
    "rule_based_classification": False,
    "safety_note": (
        "Model predictions are experimental AI outputs and must not be treated "
        "as authoritative snake-safety identification. Unknown snakes should "
        "not be approached or handled based on the model prediction."
    ),
}

EXTERNAL_TEST_REFERENCE = {
    "test_type": "Qualitative external inference test",
    "images": 7,
    "predicted_venomous": 3,
    "predicted_non_venomous": 4,
    "high_confidence": 6,
    "moderate_confidence": 1,
    "ground_truth_available": False,
    "accuracy_calculated": False,
}
