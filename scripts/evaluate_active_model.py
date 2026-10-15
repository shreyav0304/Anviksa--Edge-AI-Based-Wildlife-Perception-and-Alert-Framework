"""Evaluate both 15-class exports on the original hash-verified test images.
No training, calibration, threshold tuning, or test-set selection is performed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ui.services.evaluation import ACTIVE_ROOT, prediction_metrics
from pi_deployment.inference.tflite_inference import TFLiteWildlifeClassifier, load_class_names

def platform_path(value):
    value = value.replace(chr(92), "/")
    while "//" in value:
        value = value.replace("//", "/")
    if sys.platform != "win32" and len(value) > 2 and value[1] == ":":
        value = "/mnt/" + value[0].lower() + value[2:]
    return Path(value)

def serializable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, dict): return {k:serializable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [serializable(v) for v in value]
    return value

def write_tables(result, representation):
    for suffix,key in (("raw","raw"),("normalized","normalized")):
        with (ACTIVE_ROOT / f"confusion_matrix_{representation}_{suffix}.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=csv.writer(handle)
            writer.writerow(["Actual / Predicted", *result["classes"]])
            for name,row in zip(result["classes"],result[key],strict=True): writer.writerow([name,*row])
    with (ACTIVE_ROOT / f"per_class_metrics_{representation}.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=("name","precision","recall","f1","support"))
        writer.writeheader()
        writer.writerows(result["per_class"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args()
    classes = load_class_names()
    run = args.source_run
    final = json.loads((run / "final_evaluation.json").read_text())
    if tuple(final["provenance"]["class_order"]) != classes:
        raise ValueError("Source class order mismatch")
    checkpoint = run / "anviksa_15class_best.keras"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != final["provenance"]["checkpoint_sha256"]:
        raise ValueError("Source Keras checkpoint checksum mismatch")
    rows = [r for r in json.loads((run / "reviewed_manifest.json").read_text()) if r["split"] == "test"]
    original = [r for r in json.loads((run / "manifest.json").read_text()) if r["split"] == "test"]
    if rows != original or len(rows) != final["test_images"]:
        raise ValueError("Test rows differ from original frozen test manifest")
    labels = np.array([r["label"] for r in rows])
    paths = []
    for row in rows:
        path = platform_path(row["path"])
        if classes[row["label"]] != row["class_name"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Test image integrity mismatch: {path}")
        paths.append(path)
    with np.load(ACTIVE_ROOT / "test_predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["labels"], labels): raise ValueError("Keras cache label order mismatch")
        keras = prediction_metrics(labels, saved["probabilities"], classes)
        keras_raw = keras["raw"]
    if not np.array_equal(np.loadtxt(ACTIVE_ROOT / "confusion_matrix_keras.csv", delimiter=",", dtype=int), keras_raw):
        raise ValueError("Source Keras confusion matrix differs from labeled predictions")
    if not np.isclose(keras["accuracy"], final["keras_accuracy"], atol=1e-12, rtol=0):
        raise ValueError("Source Keras accuracy mismatch")
    summary = {"class_names":classes, "test_images":len(rows), "test_hashes_verified":len(rows),
               "source_run":run.name, "checkpoint_sha256":final["provenance"]["checkpoint_sha256"],
               "preprocessing":final["provenance"]["preprocessing"],
               "scope":"Full frozen image test; not field accuracy or independent video performance",
               "calibration":"Raw uncalibrated scores; no calibration fit on test data",
               "predictions":{}, "models":{}}
    cache = ACTIVE_ROOT / "test_predictions.npz"
    summary["predictions"]["keras"] = dict(file=cache.name, sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
    results = {"keras":serializable(keras)}
    for representation in ("float32", "float16"):
        relative = f"models/anviksa_15class_{representation}.tflite"
        model = ROOT / "pi_deployment" / relative
        classifier = TFLiteWildlifeClassifier(model, num_threads=4)
        probabilities = []
        for index, path in enumerate(paths, 1):
            probabilities.append(classifier.predict(path)["probabilities"])
            if index % 100 == 0: print(f"{representation}: {index}/{len(paths)}", flush=True)
        probabilities = np.asarray(probabilities)
        result = prediction_metrics(labels, probabilities, classes)
        cache = ACTIVE_ROOT / f"test_predictions_{representation}.npz"
        np.savez_compressed(cache, labels=labels, probabilities=probabilities)
        summary["predictions"][representation] = dict(file=cache.name, sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
        summary["models"][representation] = dict(package_path=relative, sha256=hashlib.sha256(model.read_bytes()).hexdigest(), file_size_bytes=model.stat().st_size, interpreter=classifier.interpreter_source)
        results[representation] = serializable(result)
        write_tables(result, representation)
        print(f"{representation}: {result['correct']}/{result['total']} correct; accuracy {result['accuracy']:.8%}", flush=True)
    summary["results"] = results
    (ACTIVE_ROOT / "evaluation.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print("Full export evaluation saved.", flush=True)

if __name__ == "__main__": main()
