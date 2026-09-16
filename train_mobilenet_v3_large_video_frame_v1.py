"""One controlled seven-class MobileNetV3 Large video-frame experiment.

The independent video test is not decoded until a validation-selected checkpoint
has been saved and hashed. This script never modifies source datasets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / 'clean_dataset'
CLEAN_META = ROOT / 'dataset_integrity_reports' / 'clean'
VIDEO_V2 = ROOT / 'snake_video_dataset' / 'video_dataset_candidate_v2'
RESULTS = ROOT / 'results' / 'mobilenet_v3_large_video_frame_v1'
CLASS_NAMES = ('Cow', 'Deer', 'Elephant', 'Monkey', 'Non_Venomous_Snake', 'Venomous_Snake', 'Wild_Boar')
CLEAN_FINGERPRINT = 'b103634bb6a4a01785cb7c7e3b176223325052da47228dba687edaf43a84c594'
VIDEO_FINGERPRINT = '6d84bc110af9a51f66c13ccf34766e642629b5cfc42a2aea62f81637d0866505'
SEED = 42


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def clean_rows(split: str) -> list[dict]:
    return [{
        'sample_id': f"clean:{row['relative_path']}", 'origin': 'clean_dataset', 'split': split.upper(),
        'class_name': row['class_name'], 'path': str((CLEAN / row['relative_path']).resolve()),
        'sha256': row['sha256'], 'video_id': '', 'split_group_id': '', 'species_name': '',
    } for row in read_csv(CLEAN_META / f'{split}_manifest.csv')]


def video_rows(split: str) -> list[dict]:
    output = []
    for row in read_csv(VIDEO_V2 / 'approved_frame_manifest.csv'):
        if row['final_split'] == split:
            output.append({
                'sample_id': row['review_id'], 'origin': 'video_candidate_v2', 'split': split,
                'class_name': row['verified_class'],
                'path': str((VIDEO_V2 / row['source_frame_path']).resolve()), 'sha256': row['frame_sha256'],
                'video_id': row['video_id'], 'split_group_id': row['split_group_id'],
                'species_name': row['species_name'],
            })
    return output


def counts(rows: list[dict]) -> dict[str, int]:
    counter = Counter(row['class_name'] for row in rows)
    return {name: counter[name] for name in CLASS_NAMES}


def preflight() -> dict:
    if (CLEAN_META / 'dataset_fingerprint.sha256').read_text().strip() != CLEAN_FINGERPRINT:
        raise ValueError('Clean dataset fingerprint changed')
    if (VIDEO_V2 / 'dataset_fingerprint.sha256').read_text().strip() != VIDEO_FINGERPRINT:
        raise ValueError('Video candidate v2 fingerprint changed')
    if tuple(json.loads((CLEAN_META / 'class_names.json').read_text())) != CLASS_NAMES:
        raise ValueError('Official class order changed')

    clean_train, clean_validation = clean_rows('train'), clean_rows('validation')
    add_train, add_validation = video_rows('TRAIN'), video_rows('VALIDATION')
    sealed = video_rows('INDEPENDENT_VIDEO_TEST')
    train, validation = clean_train + add_train, clean_validation + add_validation
    all_video = read_csv(VIDEO_V2 / 'approved_frame_manifest.csv')

    expected_video = {
        'TRAIN': {'Venomous_Snake': 152, 'Non_Venomous_Snake': 51},
        'VALIDATION': {'Venomous_Snake': 18, 'Non_Venomous_Snake': 11},
        'INDEPENDENT_VIDEO_TEST': {'Venomous_Snake': 11, 'Non_Venomous_Snake': 27},
    }
    actual_video = {split: Counter(row['verified_class'] for row in all_video if row['final_split'] == split)
                    for split in expected_video}
    if any(actual_video[split] != Counter(expected) for split, expected in expected_video.items()):
        raise ValueError(f'Video split counts changed: {actual_video}')
    if len(all_video) != 270:
        raise ValueError('Expected exactly 270 approved video frames')
    if any(row['final_split'] == 'INDEPENDENT_VIDEO_TEST' for row in all_video
           if row['review_id'] in {item['sample_id'] for item in train + validation}):
        raise ValueError('Independent video-test frame entered train/validation')

    train_hashes = defaultdict(list)
    validation_hashes = defaultdict(list)
    for row in train:
        train_hashes[row['sha256']].append(row['sample_id'])
    for row in validation:
        validation_hashes[row['sha256']].append(row['sample_id'])
    hash_overlap = sorted(set(train_hashes) & set(validation_hashes))
    train_videos = {row['video_id'] for row in add_train}
    validation_videos = {row['video_id'] for row in add_validation}
    train_groups = {row['split_group_id'] for row in add_train}
    validation_groups = {row['split_group_id'] for row in add_validation}
    checks = {
        'no_video_test_in_train': not ({row['video_id'] for row in sealed} & train_videos),
        'no_video_test_in_validation': not ({row['video_id'] for row in sealed} & validation_videos),
        'no_duplicate_hash_across_train_validation': not hash_overlap,
        'no_video_crossing_train_validation': not (train_videos & validation_videos),
        'no_leakage_group_crossing_train_validation': not (train_groups & validation_groups),
        'video_train_counts_exact': actual_video['TRAIN'] == Counter(expected_video['TRAIN']),
        'video_validation_counts_exact': actual_video['VALIDATION'] == Counter(expected_video['VALIDATION']),
        'video_test_counts_exact': actual_video['INDEPENDENT_VIDEO_TEST'] == Counter(expected_video['INDEPENDENT_VIDEO_TEST']),
        'all_references_exist': all(Path(row['path']).is_file() for row in train + validation + sealed),
    }
    if not all(checks.values()):
        raise ValueError(f'Pre-training integrity failure: {checks}; hash overlap={hash_overlap[:10]}')
    return {
        'clean_train_counts': counts(clean_train), 'added_video_train_counts': counts(add_train),
        'combined_train_counts': counts(train), 'clean_validation_counts': counts(clean_validation),
        'added_video_validation_counts': counts(add_validation), 'combined_validation_counts': counts(validation),
        'combined_train_total': len(train), 'combined_validation_total': len(validation),
        'sealed_video_test_total': len(sealed), 'integrity_checks': checks,
        'clean_dataset_fingerprint': CLEAN_FINGERPRINT, 'video_candidate_fingerprint': VIDEO_FINGERPRINT,
        'train_rows': train, 'validation_rows': validation, 'video_validation_rows': add_validation,
        'sealed_rows': sealed,
    }


def build_dataset(rows: list[dict], training: bool, batch_size: int = 32):
    import tensorflow as tf
    indices = {name: index for index, name in enumerate(CLASS_NAMES)}
    dataset = tf.data.Dataset.from_tensor_slices(
        ([row['path'] for row in rows], [indices[row['class_name']] for row in rows])
    )
    if training:
        dataset = dataset.shuffle(len(rows), seed=SEED, reshuffle_each_iteration=True)

    def decode(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, (224, 224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.cast(image, tf.float32), tf.one_hot(label, len(CLASS_NAMES))

    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options).map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    use_labels = labels if labels is not None else list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=use_labels, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=use_labels, average='macro', zero_division=0
    )
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(macro[0]), 'macro_recall': float(macro[1]), 'macro_f1': float(macro[2]),
        'per_class': [{
            'class': CLASS_NAMES[index], 'precision': float(precision[pos]), 'recall': float(recall[pos]),
            'f1': float(f1[pos]), 'support': int(support[pos]),
        } for pos, index in enumerate(use_labels)],
    }


def evaluate(model, rows: list[dict], result_dir: Path, name: str, labels: list[int] | None = None) -> dict:
    from sklearn.metrics import confusion_matrix
    dataset = build_dataset(rows, training=False)
    probabilities = model.predict(dataset, verbose=1)
    y_pred = probabilities.argmax(axis=1)
    indices = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
    y_true = np.asarray([indices[row['class_name']] for row in rows])
    result = metrics_for(y_true, y_pred, labels)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    np.savetxt(result_dir / f'{name}_confusion_matrix.csv', matrix, delimiter=',', fmt='%d',
               header=','.join(CLASS_NAMES), comments='')
    with (result_dir / f'{name}_predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['sample_id', 'video_id', 'expected_class', 'predicted_class', 'correct'] + [f'p_{c}' for c in CLASS_NAMES]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, prediction, probs in zip(rows, y_pred, probabilities):
            writer.writerow({'sample_id': row['sample_id'], 'video_id': row['video_id'],
                'expected_class': row['class_name'], 'predicted_class': CLASS_NAMES[prediction],
                'correct': row['class_name'] == CLASS_NAMES[prediction],
                **{f'p_{name}': float(probs[i]) for i, name in enumerate(CLASS_NAMES)}})
    result['confusion_matrix'] = matrix.tolist()
    return result


def class_weights(rows: list[dict]) -> dict[int, float]:
    counts_by_class = Counter(row['class_name'] for row in rows)
    return {i: len(rows) / (len(CLASS_NAMES) * counts_by_class[name]) for i, name in enumerate(CLASS_NAMES)}


def callbacks(path: Path, patience: int, timing: list[float]):
    import tensorflow as tf
    class Timer(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None): self.started = time.perf_counter()
        def on_epoch_end(self, epoch, logs=None): timing.append(time.perf_counter() - self.started)
    return [
        tf.keras.callbacks.ModelCheckpoint(path, monitor='val_accuracy', mode='max', save_best_only=True,
                                           save_weights_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=3, min_lr=1e-7, verbose=1),
        Timer(),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    audit = preflight()
    printable = {key: value for key, value in audit.items() if not key.endswith('_rows')}
    print(json.dumps(printable, indent=2))
    if args.preflight_only:
        return

    import tensorflow as tf
    from experiments.config import ExperimentConfig, environment_metadata, set_global_determinism
    from experiments.model_registry import build_model, compile_model, configure_fine_tuning

    set_global_determinism(SEED)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    result_dir = RESULTS / f'{timestamp}_{uuid.uuid4().hex[:8]}'
    result_dir.mkdir(parents=True, exist_ok=False)
    for name in ('train_rows', 'validation_rows', 'video_validation_rows'):
        rows = audit[name]
        with (result_dir / f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (result_dir / 'preflight_integrity.json').write_text(json.dumps(printable, indent=2) + '\n')

    config = ExperimentConfig(model_key='mobilenet_v3_large_video_frame_v1')
    exact_config = config.to_dict() | {
        'architecture_registry_key': 'mobilenet_v3_large', 'experiment_name': 'mobilenet_v3_large_video_frame_v1',
        'clean_dataset_fingerprint': CLEAN_FINGERPRINT, 'video_candidate_fingerprint': VIDEO_FINGERPRINT,
        'combined_train_count': len(audit['train_rows']), 'combined_validation_count': len(audit['validation_rows']),
        'selection_criterion': 'maximum combined validation accuracy across stage checkpoints',
        'sealed_video_test_policy': 'not decoded until selected checkpoint SHA-256 is recorded',
    }
    (result_dir / 'config.json').write_text(json.dumps(exact_config, indent=2) + '\n')
    (result_dir / 'class_names.json').write_text(json.dumps(CLASS_NAMES, indent=2) + '\n')
    environment = environment_metadata() | {'sklearn_version': __import__('sklearn').__version__}
    (result_dir / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    weights = class_weights(audit['train_rows'])
    (result_dir / 'class_weights.json').write_text(json.dumps(weights, indent=2) + '\n')

    train_ds = build_dataset(audit['train_rows'], True)
    validation_ds = build_dataset(audit['validation_rows'], False)
    bundle = build_model('mobilenet_v3_large', config, weights='imagenet')
    compile_model(bundle.model, 1e-3)
    histories, stages, durations = [], [], []

    stage1_path = result_dir / 'stage1_best.weights.h5'
    history1 = bundle.model.fit(train_ds, validation_data=validation_ds, epochs=30, class_weight=weights,
                                callbacks=callbacks(stage1_path, 6, durations))
    histories.append(history1.history); stages.extend(['stage_1'] * len(history1.history['loss']))
    stage1_epoch = int(np.argmax(history1.history['val_accuracy'])) + 1
    stage1_accuracy = float(max(history1.history['val_accuracy']))

    bundle.model.load_weights(stage1_path)
    fine_metadata = configure_fine_tuning(bundle)
    compile_model(bundle.model, 1e-5)
    fine_path = result_dir / 'fine_tuned_best.weights.h5'
    history2 = bundle.model.fit(train_ds, validation_data=validation_ds, epochs=25, class_weight=weights,
                                callbacks=callbacks(fine_path, 6, durations))
    histories.append(history2.history); stages.extend(['fine_tuning'] * len(history2.history['loss']))
    fine_epoch = int(np.argmax(history2.history['val_accuracy'])) + 1
    fine_accuracy = float(max(history2.history['val_accuracy']))
    if fine_accuracy >= stage1_accuracy:
        bundle.model.load_weights(fine_path); selected_stage = 'fine_tuning'; selected_epoch = fine_epoch
        selected_weights = fine_path; selected_accuracy = fine_accuracy
    else:
        bundle.model.load_weights(stage1_path); selected_stage = 'stage_1'; selected_epoch = stage1_epoch
        selected_weights = stage1_path; selected_accuracy = stage1_accuracy

    history_rows = []
    epoch_global = 0
    for stage, history in zip(('stage_1', 'fine_tuning'), histories):
        for i in range(len(history['loss'])):
            epoch_global += 1
            history_rows.append({'global_epoch': epoch_global, 'stage': stage, 'stage_epoch': i + 1,
                **{key: float(values[i]) for key, values in history.items()}})
    with (result_dir / 'training_history.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history_rows[0])); writer.writeheader(); writer.writerows(history_rows)

    best_model = result_dir / 'selected_model.keras'
    bundle.model.save(best_model)
    checkpoint_hash = file_sha256(best_model)
    selection = {'selected_stage': selected_stage, 'selected_stage_epoch': selected_epoch,
        'selection_criterion': 'maximum combined validation accuracy across stage checkpoints',
        'selected_validation_accuracy_during_fit': selected_accuracy, 'selected_source_weights': selected_weights.name,
        'selected_model': best_model.name, 'selected_model_sha256': checkpoint_hash,
        'hash_recorded_at_utc': datetime.now(timezone.utc).isoformat(), **fine_metadata}
    (result_dir / 'checkpoint_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    (result_dir / 'selected_model.sha256').write_text(checkpoint_hash + '  selected_model.keras\n')

    # Validation is development data and is evaluated before opening the sealed test.
    validation_metrics = evaluate(bundle.model, audit['validation_rows'], result_dir, 'combined_validation')
    video_validation_metrics = evaluate(bundle.model, audit['video_validation_rows'], result_dir,
                                        'video_validation_development', labels=[4, 5])
    (result_dir / 'validation_metrics.json').write_text(json.dumps(validation_metrics, indent=2) + '\n')
    (result_dir / 'video_validation_development_metrics.json').write_text(json.dumps(video_validation_metrics, indent=2) + '\n')

    # The checkpoint is frozen and hashed. Only now decode the sealed video test once.
    if file_sha256(best_model) != checkpoint_hash:
        raise RuntimeError('Selected checkpoint changed before sealed-test opening')
    sealed_opened_at = datetime.now(timezone.utc).isoformat()
    video_test_metrics = evaluate(bundle.model, audit['sealed_rows'], result_dir, 'independent_video_test', labels=[4, 5])
    predictions = read_csv(result_dir / 'independent_video_test_predictions.csv')
    per_video = {}
    for video_id in sorted({row['video_id'] for row in predictions}):
        rows = [row for row in predictions if row['video_id'] == video_id]
        distribution = Counter(row['predicted_class'] for row in rows)
        correct = sum(row['correct'] == 'True' for row in rows)
        source = next(row for row in audit['sealed_rows'] if row['video_id'] == video_id)
        per_video[video_id] = {'expected_class': source['class_name'], 'species': source['species_name'],
            'frames': len(rows), 'correct_frames': correct, 'incorrect_frames': len(rows) - correct,
            'accuracy': correct / len(rows), 'predicted_class_distribution': dict(sorted(distribution.items()))}
    video_test_metrics['per_video'] = per_video
    video_test_metrics['sealed_opened_at_utc'] = sealed_opened_at
    video_test_metrics['checkpoint_sha256'] = checkpoint_hash
    (result_dir / 'independent_video_test_metrics.json').write_text(json.dumps(video_test_metrics, indent=2) + '\n')

    # Evaluate the same frozen checkpoint once on the untouched official image test.
    image_test_rows = clean_rows('test')
    image_test_metrics = evaluate(bundle.model, image_test_rows, result_dir, 'frozen_image_test')
    matrix = np.asarray(image_test_metrics['confusion_matrix'])
    per_class = {row['class']: row for row in image_test_metrics['per_class']}
    image_test_metrics['snake_macro_recall'] = float(np.mean([per_class['Non_Venomous_Snake']['recall'], per_class['Venomous_Snake']['recall']]))
    image_test_metrics['snake_macro_f1'] = float(np.mean([per_class['Non_Venomous_Snake']['f1'], per_class['Venomous_Snake']['f1']]))
    image_test_metrics['nonvenomous_to_venomous_errors'] = int(matrix[4, 5])
    image_test_metrics['venomous_to_nonvenomous_errors'] = int(matrix[5, 4])
    (result_dir / 'frozen_image_test_metrics.json').write_text(json.dumps(image_test_metrics, indent=2) + '\n')

    comparison = {
        'official_v1': {'accuracy': .9541284403669725, 'macro_precision': .9688672743760461,
            'macro_recall': .9686971396837043, 'macro_f1': .9687245232484056,
            'snake_macro_recall': .9215702570921986, 'snake_macro_f1': .9237181985191782,
            'nonvenomous_to_venomous_errors': 11, 'venomous_to_nonvenomous_errors': 8},
        'experimental_video_frame_v1': {key: image_test_metrics[key] for key in (
            'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'snake_macro_recall', 'snake_macro_f1',
            'nonvenomous_to_venomous_errors', 'venomous_to_nonvenomous_errors')},
        'promotion_status': 'NOT_PROMOTED',
    }
    (result_dir / 'v1_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    artifact_hashes = {path.name: file_sha256(path) for path in result_dir.iterdir() if path.is_file()}
    (result_dir / 'artifact_hashes.json').write_text(json.dumps(artifact_hashes, indent=2) + '\n')
    print('EXPERIMENT_RESULT_DIR=' + str(result_dir.resolve()))


if __name__ == '__main__':
    main()
