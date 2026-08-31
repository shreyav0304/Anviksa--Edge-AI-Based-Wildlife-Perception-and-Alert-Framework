from pathlib import Path
import sys

from inference_engine import WildlifeInferenceEngine


def main() -> int:
    """Run backward-compatible single-image prediction from the command line."""
    if len(sys.argv) != 2:
        print("\nUsage:")
        print('python test_model.py "/path/to/image.jpg"')
        return 2

    image_path = Path(sys.argv[1])

    try:
        print("\nLoading Anviksa AI model...")
        engine = WildlifeInferenceEngine()
        print("Model loaded successfully.")
        print("Classes:", list(engine.class_names))
        result = engine.predict(image_path)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"\nERROR: {exc}")
        return 1

    print("\n" + "=" * 60)
    print("ANVIKSA AI PREDICTION")
    print("=" * 60)
    print(f"\nImage            : {image_path.name}")
    print(f"Predicted Class  : {result['predicted_class']}")
    print(f"Confidence       : {result['confidence'] * 100:.2f}%")
    print(f"Inference Time   : {result['inference_time_ms']:.2f} ms")
    print(f"Processing Time  : {result['processing_time_ms']:.2f} ms")

    print("\nAll Class Probabilities:")
    print("-" * 60)
    for class_name, probability in result["probabilities"].items():
        print(f"{class_name:<25} : {probability * 100:6.2f}%")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
