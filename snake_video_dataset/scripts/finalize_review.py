"""Build an admission manifest from explicit frame review decisions; never copy to clean_dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

CLASSES = ('Non_Venomous_Snake', 'Venomous_Snake')


def rows(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=Path(__file__).resolve().parents[1] / 'video_dataset_candidate_v1')
    parser.add_argument('--decisions', type=Path, required=True, help='CSV worklist with video_id,frame_number,review_status,verified_class,verification_note')
    args = parser.parse_args()
    videos = {v['video_id']: v for v in rows(args.candidate / 'video_manifest.csv')}
    frames = {(f['video_id'], f['frame_number']): f for f in rows(args.candidate / 'review_index.csv')}
    admitted = []
    seen = set()
    for decision in rows(args.decisions):
        key = (decision['video_id'], decision['frame_number'])
        if key in seen:
            raise ValueError(f'Duplicate review decision: {key}')
        seen.add(key)
        if key not in frames:
            raise ValueError(f'Unknown frame: {key}')
        if decision['review_status'] not in ('APPROVED', 'REJECTED', 'REVIEW_REQUIRED'):
            raise ValueError(f'Invalid review status: {key}')
        if decision['review_status'] != 'APPROVED':
            continue
        frame = frames[key]
        video = videos[key[0]]
        label = decision.get('verified_class') or decision.get('verified_label')
        if label not in CLASSES or not decision['verification_note'].strip():
            raise ValueError(f'Approved frame needs a verified snake label and evidence note: {key}')
        if not frame['frame_path'] or frame['quality_status'] == 'REJECTED' or frame['duplicate_status'] != 'UNIQUE':
            raise ValueError(f'Cannot admit rejected or duplicate frame: {key}')
        if video['class_label'] not in ('UNKNOWN', label):
            raise ValueError(f'Frame label conflicts with source video label: {key}')
        actual = hashlib.sha256((args.candidate / frame['frame_path']).read_bytes()).hexdigest()
        if actual != frame['frame_sha256']:
            raise ValueError(f'Candidate frame content changed: {key}')
        admitted.append(dict(video_id=key[0], group_id=video['group_id'], split=video['assigned_split'],
            frame_number=key[1], timestamp_seconds=frame['timestamp_seconds'], class_label=label,
            species_name=decision.get('species_name', ''), relative_path=frame['frame_path'], frame_sha256=actual, verification_note=decision['verification_note']))
    group_splits = {}
    for row in admitted:
        previous = group_splits.setdefault(row['group_id'], row['split'])
        if previous != row['split']:
            raise ValueError(f'Group leakage: {row["group_id"]}')
    admitted.sort(key=lambda row: (row['video_id'], int(row['frame_number'])))
    canonical = json.dumps(admitted, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    fingerprint = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    path = args.candidate / 'admitted_manifest.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['video_id','group_id','split','frame_number','timestamp_seconds','class_label','species_name','relative_path','frame_sha256','verification_note'])
        writer.writeheader()
        writer.writerows(admitted)
    (args.candidate / 'admitted_fingerprint.sha256').write_text(fingerprint + '\n', encoding='utf-8')
    print(json.dumps({'admitted_frames': len(admitted), 'per_class': dict(Counter(r['class_label'] for r in admitted)),
        'per_split': dict(Counter(r['split'] for r in admitted)), 'unassigned_approved_frames': sum(r['split'] == 'UNASSIGNED_REVIEW' for r in admitted),
        'admitted_fingerprint': fingerprint}, indent=2))


if __name__ == '__main__':
    main()
