"""Audit source videos and create reviewable, group-safe frame candidates.

No output is admitted to training until review_index.csv has review_status=APPROVED.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('Cow', 'Deer', 'Elephant', 'Monkey', 'Non_Venomous_Snake', 'Venomous_Snake', 'Wild_Boar')
EXTENSIONS = {'.mp4', '.webm', '.avi', '.mov', '.mkv', '.m4v'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def existing_metadata() -> dict[str, dict]:
    path = ROOT / 'metadata' / 'global_video_manifest.csv'
    if not path.exists():
        return {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return {row['relative_path'].replace('\\', '/'): row for row in csv.DictReader(stream)}


def group_overrides() -> dict[str, str]:
    path = ROOT / 'metadata' / 'group_overrides.csv'
    if not path.exists():
        return {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return {row['relative_path'].replace('\\', '/'): row['group_id'] for row in csv.DictReader(stream) if row['group_id']}


def category(path: Path) -> str:
    try:
        part = path.relative_to(ROOT / 'source_videos').parts[0].lower()
    except ValueError:
        return 'external'
    return {'user_collected': 'rescuer', 'web_collected': 'web'}.get(part, part)


def inspected_label(old: dict) -> tuple[str, str]:
    label = old.get('current_class', '')
    if old.get('verification_status') == 'VERIFIED' and label in CLASSES:
        return label, 'SOURCE_VERIFIED_FRAME_REVIEW_REQUIRED'
    return 'UNKNOWN', 'REVIEW_REQUIRED'


def choose_split(groups: dict[str, list[dict]]) -> None:
    # Reserve validation and video test only where a class has enough independent,
    # verified recording groups; unverified groups receive no split.
    by_class = defaultdict(list)
    for group, videos in groups.items():
        labels = {v['class_label'] for v in videos}
        if len(labels) == 1 and 'UNKNOWN' not in labels:
            by_class[next(iter(labels))].append(group)
    assignments = {}
    for label, ids in by_class.items():
        ids.sort(key=lambda g: hashlib.sha256(g.encode()).hexdigest())
        for group in ids:
            assignments[group] = 'TRAIN'
        if len(ids) >= 5:
            assignments[ids[0]] = 'VALIDATION'
        if len(ids) >= 8:
            assignments[ids[1]] = 'INDEPENDENT_VIDEO_TEST'
    for group, videos in groups.items():
        split = assignments.get(group, 'UNASSIGNED_REVIEW')
        for video in videos:
            video['assigned_split'] = split


def quality(frame: np.ndarray, args) -> tuple[str, str]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < args.black_mean and np.percentile(gray, 95) < args.black_p95:
        return 'REJECTED', 'NEAR_BLACK'
    if gray.mean() > args.white_mean and np.percentile(gray, 5) > args.white_p5:
        return 'REJECTED', 'SEVERELY_OVEREXPOSED'
    if cv2.Laplacian(cv2.resize(gray, (160, 90)), cv2.CV_64F).var() < args.blur_variance:
        return 'REVIEW', 'VERY_LOW_DETAIL_OR_BLUR'
    return 'PASS', ''


def signature(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)


def sampling_due(index: int, fps: float, last_index: int, previous: np.ndarray | None, current: np.ndarray, args) -> bool:
    if index == 0:
        return True
    if args.method == 'interval':
        return index - last_index >= max(1, round(args.interval_seconds * fps))
    if args.method == 'fps':
        return index - last_index >= max(1, round(fps / args.target_fps))
    elapsed = (index - last_index) / fps
    if elapsed < args.min_seconds:
        return False
    change = float(np.mean(np.abs(current - previous))) if previous is not None else 255.0
    return change >= args.min_frame_change or elapsed >= args.max_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'source_videos')
    parser.add_argument('--output', type=Path, default=ROOT / 'video_dataset_candidate_v1')
    parser.add_argument('--source', help='Override source category for a single input video')
    parser.add_argument('--label', choices=CLASSES, help='Provisional label only; still requires manual frame review')
    parser.add_argument('--group-id', help='Recording/session ID for a single input video')
    parser.add_argument('--method', choices=('interval', 'fps', 'motion'), default='interval')
    parser.add_argument('--interval-seconds', type=float, default=2.0)
    parser.add_argument('--target-fps', type=float, default=1.0)
    parser.add_argument('--min-frame-change', type=float, default=9.0)
    parser.add_argument('--min-seconds', type=float, default=0.5)
    parser.add_argument('--max-seconds', type=float, default=4.0)
    parser.add_argument('--duplicate-difference', type=float, default=3.0)
    parser.add_argument('--black-mean', type=float, default=3.0)
    parser.add_argument('--black-p95', type=float, default=8.0)
    parser.add_argument('--white-mean', type=float, default=252.0)
    parser.add_argument('--white-p5', type=float, default=245.0)
    parser.add_argument('--blur-variance', type=float, default=2.0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if min(args.interval_seconds, args.target_fps, args.min_seconds, args.max_seconds) <= 0:
        parser.error('Sampling intervals and FPS must be positive')
    files = [args.input] if args.input.is_file() else sorted(p for p in args.input.rglob('*') if p.suffix.lower() in EXTENSIONS)
    old_rows = existing_metadata()
    overrides = group_overrides()
    videos = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
        old = old_rows.get(rel, {})
        digest = sha256(path)
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        readable = cap.isOpened() and fps > 0 and frames > 0
        cap.release()
        label, verification = inspected_label(old)
        if args.label:
            label, verification = args.label, 'PROVISIONAL_FRAME_REVIEW_REQUIRED'
        videos.append(dict(video_id='VID_' + digest[:16], original_filename=path.name,
            relative_path=rel, source_category=args.source or category(path),
            source_owner_note=old.get('notes') or old.get('source_type') or 'UNKNOWN',
            group_id=args.group_id or overrides.get(rel) or ((category(path) + '_' + old['source_group_id']) if old.get('source_group_id') else 'GROUP_' + digest[:16]),
            class_label=label, venom_status=('VENOMOUS' if label == 'Venomous_Snake' else 'NON_VENOMOUS' if label == 'Non_Venomous_Snake' else 'UNKNOWN'),
            species_name=old.get('species_scientific_name') if verification.startswith('SOURCE_VERIFIED') else 'UNKNOWN',
            verification_status=verification, duration_seconds=round(frames / fps, 3) if readable else 0,
            fps=round(fps, 5), width=width, height=height, total_frames=frames,
            file_sha256=digest, readable=readable, assigned_split='', method=args.method,
            extraction_settings=json.dumps({k: getattr(args, k) for k in ('interval_seconds','target_fps','min_frame_change','min_seconds','max_seconds','duplicate_difference','black_mean','black_p95','white_mean','white_p5','blur_variance')}, sort_keys=True),
            sampled_frames=0, duplicates_removed=0, quality_rejected=0, manual_review_required=0,
            decode_failures=0, _path=path))
    groups = defaultdict(list)
    for video in videos:
        groups[video['group_id']].append(video)
    choose_split(groups)
    out = args.output
    review = []
    if not args.dry_run:
        (out / 'frames').mkdir(parents=True, exist_ok=True)
    for video in videos:
        if not video['readable']:
            continue
        cap = cv2.VideoCapture(str(video['_path']))
        index = 0
        last_index = -10**9
        last_sample = None
        last_kept = None
        while True:
            ok, frame = cap.read()
            if not ok:
                if index < video['total_frames'] - 1:
                    video['decode_failures'] += 1
                break
            sig = signature(frame)
            if sampling_due(index, video['fps'], last_index, last_sample, sig, args):
                last_index, last_sample = index, sig
                video['sampled_frames'] += 1
                status, reason = quality(frame, args)
                duplicate = 'UNIQUE'
                if last_kept is not None and float(np.mean(np.abs(sig - last_kept))) < args.duplicate_difference:
                    duplicate = 'NEAR_DUPLICATE'
                    video['duplicates_removed'] += 1
                elif status != 'REJECTED':
                    last_kept = sig
                if status == 'REJECTED':
                    video['quality_rejected'] += 1
                else:
                    video['manual_review_required'] += 1 if duplicate == 'UNIQUE' else 0
                frame_rel = f"frames/{video['video_id']}/{index:08d}.jpg"
                frame_hash = ''
                if not args.dry_run and status != 'REJECTED' and duplicate == 'UNIQUE':
                    destination = out / frame_rel
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                        raise OSError(f'Could not save {destination}')
                    frame_hash = sha256(destination)
                review.append(dict(video_id=video['video_id'], frame_number=index,
                    timestamp_seconds=round(index / video['fps'], 3), frame_path=frame_rel if status != 'REJECTED' and duplicate == 'UNIQUE' else '', frame_sha256=frame_hash,
                    current_label=video['class_label'], source=video['source_category'], group_id=video['group_id'],
                    assigned_split=video['assigned_split'], quality_status=status, quality_reason=reason,
                    duplicate_status=duplicate, review_status='REVIEW_REQUIRED'))
            index += 1
        cap.release()
    fields = [k for k in videos[0] if k != '_path'] if videos else []
    rows = [{k: v[k] for k in fields} for v in videos]
    review_fields = ['video_id','frame_number','timestamp_seconds','frame_path','frame_sha256','current_label','source','group_id','assigned_split','quality_status','quality_reason','duplicate_status','review_status']
    canonical = json.dumps({'videos': rows, 'frames': review}, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    fingerprint = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    report = dict(videos_discovered=len(videos), video_sources=dict(Counter(v['source_category'] for v in videos)),
        groups_discovered=len(groups), total_duration_seconds=round(sum(v['duration_seconds'] for v in videos), 3),
        raw_frame_count=sum(v['total_frames'] for v in videos), frames_sampled=len(review),
        frames_rejected_for_quality=sum(v['quality_rejected'] for v in videos),
        near_duplicates_removed=sum(v['duplicates_removed'] for v in videos),
        frames_requiring_manual_review=sum(v['manual_review_required'] for v in videos),
        verified_usable_frames_per_class={}, train_validation_video_test_distribution=dict(Counter(v['assigned_split'] for v in videos)),
        leakage_checks={'groups_with_multiple_splits': [g for g, vs in groups.items() if len({v['assigned_split'] for v in vs}) > 1]},
        candidate_dataset_path=str(out.resolve()), candidate_manifest_path=str((out / 'video_manifest.csv').resolve()),
        dataset_fingerprint=fingerprint, dry_run=args.dry_run)
    if not args.dry_run:
        write_csv(out / 'video_manifest.csv', rows, fields)
        write_csv(out / 'review_index.csv', review, review_fields)
        (out / 'dataset_fingerprint.sha256').write_text(fingerprint + '\n', encoding='utf-8')
        (out / 'audit_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
