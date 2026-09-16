"""Summarize authoritative manual frame decisions without changing dataset files."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'video_dataset_candidate_v1'
STATUSES = ('APPROVED', 'REJECTED', 'REVIEW_REQUIRED')
CLASSES = ('Venomous_Snake', 'Non_Venomous_Snake', 'UNKNOWN')


def load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def status_counts(frames: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(frame['review_status'] for frame in frames)
    return {status.lower(): counts[status] for status in STATUSES}


def grouped_counts(frames: list[dict[str, str]], field: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for frame in frames:
        grouped[frame[field]][frame['review_status']] += 1
    return {
        key: {'total': sum(counts.values()), **{status.lower(): counts[status] for status in STATUSES}}
        for key, counts in sorted(grouped.items())
    }


def subset_summary(frames: list[dict[str, str]], detailed: bool = False) -> dict:
    summary = {'total': len(frames), **status_counts(frames)}
    if detailed:
        approved = Counter(
            frame['verified_class'] for frame in frames if frame['review_status'] == 'APPROVED'
        )
        summary.update(
            approved_venomous_snake=approved['Venomous_Snake'],
            approved_non_venomous_snake=approved['Non_Venomous_Snake'],
            counts_by_video=grouped_counts(frames, 'video_id'),
            counts_by_recording_group=grouped_counts(frames, 'group_id'),
            counts_by_source=grouped_counts(frames, 'source'),
            counts_by_provisional_split=grouped_counts(frames, 'split'),
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--worklist', type=Path,
        help='Explicit review CSV. By default review_worklist_manual.csv is authoritative when it exists.',
    )
    args = parser.parse_args()
    manual = ROOT / 'review_worklist_manual.csv'
    source = ROOT / 'review_worklist.csv'
    worklist = args.worklist or (manual if manual.exists() else source)

    frames = load(worklist)
    original = load(source)
    videos = {video['video_id']: video for video in load(ROOT / 'video_manifest.csv')}
    audit = {video['video_id']: video for video in load(ROOT / 'video_label_audit.csv')}
    proposed = load(ROOT / 'proposed_group_review.csv')

    original_ids = {(frame['video_id'], frame['frame_number']) for frame in original}
    ids: set[tuple[str, str]] = set()
    for frame in frames:
        key = (frame['video_id'], frame['frame_number'])
        if key in ids or key not in original_ids or key[0] not in videos or key[0] not in audit:
            raise ValueError(f'Duplicate or unknown frame: {key}')
        ids.add(key)
        if frame['review_status'] not in STATUSES or frame['verified_class'] not in CLASSES:
            raise ValueError(f'Invalid review status or class: {key}')
        if frame['review_status'] == 'APPROVED' and frame['verified_class'] == 'UNKNOWN':
            raise ValueError(f'Approved frame has unknown class: {key}')
    if ids != original_ids:
        raise ValueError(f'Authoritative worklist does not contain all candidate frames: {len(ids)} of {len(original_ids)}')

    verified_ids = {video_id for video_id, video in audit.items() if video['label_status'] == 'SOURCE_VERIFIED'}
    verified = [frame for frame in frames if frame['video_id'] in verified_ids]
    unverified = [frame for frame in frames if frame['video_id'] not in verified_ids]
    if len(frames) != 608:
        raise ValueError(f'Expected 608 candidate frames, found {len(frames)}')
    if len(verified) != 281:
        raise ValueError(f'Expected 281 SOURCE_VERIFIED frames, found {len(verified)}')
    verified_counts = status_counts(verified)
    if sum(verified_counts.values()) != 281:
        raise ValueError('SOURCE_VERIFIED approved + rejected + review_required does not equal 281')

    unresolved = sorted(
        video_id for video_id, video in audit.items() if video['label_status'] != 'SOURCE_VERIFIED'
    )
    group_splits: dict[str, set[str]] = defaultdict(set)
    for video in videos.values():
        group_splits[video['group_id']].add(video['assigned_split'])

    report = {
        'authoritative_review_source': str(worklist.resolve()),
        'all_candidate_frames': subset_summary(frames),
        'source_verified_subset': subset_summary(verified, detailed=True),
        'unverified_subset': subset_summary(unverified),
        'unresolved_label_videos': unresolved,
        'potential_group_conflicts': [row for row in proposed if row['possible_related_video']],
        'groups_crossing_splits': {
            group: sorted(splits) for group, splits in sorted(group_splits.items()) if len(splits) > 1
        },
        'validation': {
            'source_verified_total_is_281': len(verified) == 281,
            'source_verified_status_sum_is_281': sum(verified_counts.values()) == 281,
            'all_candidate_total_is_608': len(frames) == 608,
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
