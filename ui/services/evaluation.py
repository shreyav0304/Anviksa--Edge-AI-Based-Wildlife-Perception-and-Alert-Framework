"""Recompute measured metrics from labeled predictions, without fitting to test data."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ACTIVE_ROOT = ROOT / "results/deployment/mobilenet_v3_large_15class"

def prediction_metrics(labels, probabilities, classes):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    count, size = len(labels), len(classes)
    if count == 0 or len(set(classes)) != size or size < 2:
        raise ValueError("Evaluation needs samples and unique class names")
    if labels.shape != (count,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Invalid ground-truth labels")
    if probabilities.shape != (count, size) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid prediction shape or nonfinite probabilities")
    if np.any(labels < 0) or np.any(labels >= size) or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Labels or probabilities outside their valid range")
    if not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError("Prediction rows do not sum to one")
    predicted = probabilities.argmax(axis=1)
    raw = np.zeros((size, size), dtype=np.int64)
    np.add.at(raw, (labels, predicted), 1)
    support = raw.sum(axis=1)
    true_positive = raw.diagonal()
    precision = np.divide(true_positive, raw.sum(axis=0), out=np.zeros(size), where=raw.sum(axis=0) != 0)
    recall = np.divide(true_positive, support, out=np.zeros(size), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(size), where=precision + recall != 0)
    normalized = np.divide(raw, support[:, None], out=np.zeros((size, size)), where=support[:, None] != 0)
    correct = predicted == labels
    scores = probabilities.max(axis=1)
    accuracy = float(correct.mean())
    # Wilson interval: sampling uncertainty on this split, not field performance.
    z = 1.959963984540054
    denominator = 1 + z*z/count
    center = (accuracy + z*z/(2*count))/denominator
    half = z * math.sqrt(accuracy*(1-accuracy)/count + z*z/(4*count*count))/denominator
    ece = 0.0
    bins = []
    for index in range(15):
        mask = (scores >= index/15) & ((scores <= 1) if index == 14 else (scores < (index+1)/15))
        n = int(mask.sum())
        if n:
            bin_accuracy, bin_score = float(correct[mask].mean()), float(scores[mask].mean())
            ece += n/count * abs(bin_accuracy-bin_score)
            bins.append(dict(lower=index/15, upper=(index+1)/15, count=n, accuracy=bin_accuracy, mean_score=bin_score))
    one_hot = np.eye(size)[labels]
    return dict(classes=tuple(classes), raw=raw, normalized=normalized, total=count,
                correct=int(correct.sum()), accuracy=accuracy, accuracy_interval=(center-half, center+half),
                macro_precision=float(precision.mean()), macro_recall=float(recall.mean()), macro_f1=float(f1.mean()),
                weighted_f1=float(np.average(f1, weights=support)), mse=float(np.mean((probabilities-one_hot)**2)),
                mean_confidence=float(scores.mean()), median_confidence=float(np.median(scores)),
                error_mean_confidence=float(scores[~correct].mean()) if np.any(~correct) else None,
                errors_above_99=int(((scores >= .99) & ~correct).sum()), saturated_scores=int((scores == 1).sum()),
                ece=ece, reliability_bins=bins,
                per_class=[dict(name=name, precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]), support=int(support[i])) for i,name in enumerate(classes)])

def load_active_evaluation(representation="float32"):
    metadata = json.loads((ACTIVE_ROOT / "evaluation.json").read_text(encoding="utf-8"))
    entry = metadata["predictions"][representation]
    path = ACTIVE_ROOT / entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Active evaluation prediction checksum mismatch")
    with np.load(path, allow_pickle=False) as saved:
        measured = prediction_metrics(saved["labels"], saved["probabilities"], metadata["class_names"])
    if measured["total"] != metadata["test_images"]:
        raise ValueError("Active evaluation sample count mismatch")
    from pi_deployment.inference.tflite_inference import load_class_names
    if measured["classes"] != load_class_names():
        raise ValueError("Evaluation classes differ from deployed model")
    if representation != "keras":
        model = ROOT / "pi_deployment" / metadata["models"][representation]["package_path"]
        if hashlib.sha256(model.read_bytes()).hexdigest() != metadata["models"][representation]["sha256"]:
            raise ValueError("Evaluation model checksum mismatch")
    return {**measured, "metadata": metadata, "representation": representation}

def format_model_score(value):
    """Do not round a near-one score into a claim of perfect certainty."""
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Model score must be finite and in [0, 1]")
    if value == 1:
        return "1.0 (saturated score)"
    if value == 0:
        return "0%"
    rounded = f"{value*100:.6f}"
    if rounded == "100.000000":
        return ">99.999999%"
    if rounded == "0.000000":
        return "<0.000001%"
    return rounded.rstrip("0").rstrip(".") + "%"
