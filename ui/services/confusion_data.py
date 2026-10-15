"""Confusion matrices recomputed from labeled predictions for each evaluation."""
from __future__ import annotations
import csv
from dataclasses import dataclass
import numpy as np
from .comparison_data import COMPLETED_RUNS, PROJECT_ROOT
from .evaluation import prediction_metrics, load_active_evaluation

CLASS_NAMES = ("Cow", "Deer", "Elephant", "Monkey", "Non_Venomous_Snake", "Venomous_Snake", "Wild_Boar")
ACTIVE_MODEL = "MobileNetV3 Large - 15 classes (Float32)"

@dataclass(frozen=True)
class ConfusionMatrixRecord:
    model: str
    raw: np.ndarray
    normalized: np.ndarray
    metrics: dict[str, float]
    total: int
    correct: int
    incorrect: int
    matrix_accuracy: float
    largest_confusion: tuple[str, str, int]
    nonvenomous_to_venomous: int
    venomous_to_nonvenomous: int
    interpretation: str
    class_names: tuple[str, ...] = CLASS_NAMES
    evaluation_scope: str = "Historical 7-class frozen image test; 545 images"

def _record(name, measured, scope):
    classes, raw = measured["classes"], measured["raw"]
    nv, v = classes.index("Non_Venomous_Snake"), classes.index("Venomous_Snake")
    off = raw.copy()
    np.fill_diagonal(off, 0)
    row, column = np.unravel_index(int(off.argmax()), off.shape)
    metrics = {k:measured[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
    metrics["venomous_recall"] = measured["per_class"][v]["recall"]
    metrics["snake_macro_recall"] = (measured["per_class"][nv]["recall"] + metrics["venomous_recall"]) / 2
    errors = measured["total"] - measured["correct"]
    return ConfusionMatrixRecord(name, raw, measured["normalized"], metrics, measured["total"], measured["correct"], errors,
        measured["accuracy"], (classes[row],classes[column],int(off[row,column])), int(raw[nv,v]),int(raw[v,nv]),
        f"{measured['correct']}/{measured['total']} correct; {errors} errors. "
        f"Snake-type confusions: {raw[nv,v]} non-venomous predicted venomous; {raw[v,nv]} venomous predicted non-venomous. "
        "Values describe this labeled image split, not field accuracy. Small class supports can produce perfect rows without proving perfect performance.",
        classes, scope)

def load_confusion_records():
    active = load_active_evaluation()
    # Confirm the cached computation also agrees with the packaged audit report.
    if not np.array_equal(active["raw"], active["metadata"]["results"]["float32"]["raw"]):
        raise ValueError("Active matrix differs from the full export evaluation report")
    records = {ACTIVE_MODEL:_record(ACTIVE_MODEL, active, "Active 15-class Float32 export; 670 hash-verified frozen test images")}
    with (PROJECT_ROOT / "results/comparison/final_valid_model_comparison.csv").open(newline="",encoding="utf-8") as f:
        summaries={r["Model"]:r for r in csv.DictReader(f)}
    reference_labels = None
    for name, run in COMPLETED_RUNS.items():
        slug={"Custom CNN":"custom_cnn","MobileNetV2":"mobilenet_v2","MobileNetV3 Large":"mobilenet_v3_large"}[name]
        path=PROJECT_ROOT / "results/model_comparison_dashboard" / f"prediction_probabilities_{slug}.npz"
        with np.load(path,allow_pickle=False) as saved:
            labels, probabilities = saved["true_labels"], saved["probabilities"]
        if reference_labels is not None and not np.array_equal(labels,reference_labels):
            raise ValueError("Historical models do not share the same test labels")
        reference_labels=labels.copy()
        measured=prediction_metrics(labels,probabilities,CLASS_NAMES)
        if measured["total"] != 545:
            raise ValueError("Historical frozen test sample count mismatch")
        record=_record(name,measured,"Historical 7-class frozen image test; 545 images; separate from active 15-class evaluation")
        for metric, field in (("accuracy","Test Accuracy"),("macro_precision","Macro Precision"),("macro_recall","Macro Recall"),("macro_f1","Macro F1"),("venomous_recall","Venomous Snake Recall"),("snake_macro_recall","Snake Macro Recall")):
            if not np.isclose(record.metrics[metric],float(summaries[name][field]),rtol=0,atol=1e-9):
                raise ValueError(f"{name}: recomputed {metric} differs from published summary")
        # Optional private run matrices must agree too; they never override predictions.
        raw_path=run / "confusion_matrix_raw.csv"
        if raw_path.is_file():
            with raw_path.open(newline="",encoding="utf-8") as f: rows=list(csv.reader(f))
            if tuple(rows[0][1:]) != CLASS_NAMES or tuple(r[0] for r in rows[1:]) != CLASS_NAMES:
                raise ValueError(f"{name}: historical class order mismatch")
            if not np.array_equal(np.array([r[1:] for r in rows[1:]],dtype=int),record.raw):
                raise ValueError(f"{name}: historical raw matrix differs from predictions")
        records[name]=record
    return records
