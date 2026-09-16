"""Apply owner-confirmed video labels to frame worklist without approving frames."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'video_dataset_candidate_v1'
LABELS = {'Venomous_Snake', 'Non_Venomous_Snake'}


def load(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--worklist', type=Path, default=ROOT / 'review_worklist.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'review_worklist_with_video_labels.csv')
    args = parser.parse_args()
    source = {v['video_id']: v for v in load(ROOT / 'video_manifest.csv')}
    decisions = {}
    for row in load(args.decisions):
        label = row['verified_class'].strip()
        if label == 'UNKNOWN' or not label:
            continue
        if row['video_id'] in decisions or row['video_id'] not in source or label not in LABELS or not row['verification_note'].strip():
            raise ValueError(f'Invalid, duplicate, or unsupported video confirmation: {row}')
        current = source[row['video_id']]['class_label']
        if current in LABELS and current != label:
            raise ValueError(f'Confirmed label conflicts with source metadata for {row["video_id"]}; resolve provenance first')
        decisions[row['video_id']] = row
    worklist = load(args.worklist)
    for frame in worklist:
        if decision := decisions.get(frame['video_id']):
            frame['verified_class'] = decision['verified_class']
            frame['species_name'] = decision['species_name']
            frame['verification_note'] = decision['verification_note']
            # Deliberately preserve individual frame status, rejection reasons and notes.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(worklist[0]))
        writer.writeheader()
        writer.writerows(worklist)
    print(f'Applied labels for {len(decisions)} videos; approved no frames; wrote {args.output}')


if __name__ == '__main__':
    main()
