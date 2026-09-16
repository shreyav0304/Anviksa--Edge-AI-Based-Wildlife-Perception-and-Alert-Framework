"""Freeze the reviewed, group-aware snake-video candidate without copying images."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'video_dataset_candidate_v1'
OUTPUT = ROOT / 'video_dataset_candidate_v2'
MANUAL = SOURCE / 'review_worklist_manual.csv'

SPLITS = {
    'VALIDATION': {
        'VID_78328983a941f244',  # Elaphe climacophora, web, nonvenomous
        'VID_c6dec41a33140417',  # Aspidelaps lubricus, rescuer, venomous
        'VID_cf668d0677dbbd8f',  # Crotalus horridus, web, venomous
    },
    'INDEPENDENT_VIDEO_TEST': {
        'VID_babc3e7bef3c3d3e',  # Python regius, web, nonvenomous
        'VID_2fa5a241b315c1a7',  # Bitis nasicornis, rescuer, venomous
    },
}
MERGED_GROUPS = {
    'VID_163687650fcdfb63': 'LEAKAGE_GROUP_STV1747_0001',
    'VID_7871d82c6a5700eb': 'LEAKAGE_GROUP_STV1747_0001',
    'VID_7c9fedb83b820769': 'LEAKAGE_GROUP_STV1608_0010',
    'VID_ffd2ba7c4edda0a1': 'LEAKAGE_GROUP_STV1608_0010',
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_for(video_id: str) -> str:
    for split, members in SPLITS.items():
        if video_id in members:
            return split
    return 'TRAIN'


def summarize(rows: list[dict], split_rows: list[dict]) -> dict:
    result = {}
    for split in ('TRAIN', 'VALIDATION', 'INDEPENDENT_VIDEO_TEST'):
        frames = [row for row in rows if row['final_split'] == split]
        videos = [row for row in split_rows if row['final_split'] == split]
        result[split] = {
            'videos': len(videos),
            'split_groups': len({row['split_group_id'] for row in videos}),
            'approved_frames': len(frames),
            'class_frames': dict(sorted(Counter(row['verified_class'] for row in frames).items())),
            'source_frames': dict(sorted(Counter(row['source'] for row in frames).items())),
            'species': sorted({row['species_name'] for row in frames}),
            'video_ids': sorted(row['video_id'] for row in videos),
            'split_group_ids': sorted({row['split_group_id'] for row in videos}),
        }
    return result


def main() -> None:
    manual_hash_before = hashlib.sha256(MANUAL.read_bytes()).hexdigest()
    manual = read(MANUAL)
    audit = {row['video_id']: row for row in read(SOURCE / 'video_label_audit.csv')}
    videos = {row['video_id']: row for row in read(SOURCE / 'video_manifest.csv')}
    review_index = {
        f"{row['video_id']}:{int(row['frame_number']):08d}": row
        for row in read(SOURCE / 'review_index.csv') if row['frame_path']
    }

    eligible = [row for row in manual if audit[row['video_id']]['label_status'] == 'SOURCE_VERIFIED'
                and row['review_status'] == 'APPROVED']
    rejected = [row for row in manual if audit[row['video_id']]['label_status'] == 'SOURCE_VERIFIED'
                and row['review_status'] == 'REJECTED']
    unresolved = [row for row in manual if audit[row['video_id']]['label_status'] != 'SOURCE_VERIFIED']
    verified_videos = sorted({row['video_id'] for row in eligible})
    if (len(eligible), Counter(row['verified_class'] for row in eligible), len(rejected), len(unresolved)) != (
        270, Counter({'Venomous_Snake': 181, 'Non_Venomous_Snake': 89}), 11, 327
    ):
        raise ValueError('Authoritative review totals do not match the frozen-candidate requirements')

    split_manifest = []
    for video_id in verified_videos:
        audit_row = audit[video_id]
        split_group = MERGED_GROUPS.get(video_id, 'LEAKAGE_' + videos[video_id]['group_id'])
        split_manifest.append({
            'video_id': video_id,
            'original_filename': audit_row['original_filename'],
            'species_name': audit_row['species_label'],
            'verified_class': audit_row['current_snake_class'],
            'source': audit_row['source'],
            'original_group_id': audit_row['group_session_id'],
            'split_group_id': split_group,
            'approved_frame_count': sum(row['video_id'] == video_id for row in eligible),
            'final_split': split_for(video_id),
        })

    approved_manifest = []
    for row in eligible:
        indexed = review_index[row['review_id']]
        audit_row = audit[row['video_id']]
        approved_manifest.append({
            'review_id': row['review_id'],
            'video_id': row['video_id'],
            'frame_number': row['frame_number'],
            'timestamp_seconds': row['timestamp_seconds'],
            'verified_class': row['verified_class'],
            'species_name': audit_row['species_label'],
            'source': row['source'],
            'original_group_id': row['group_id'],
            'split_group_id': MERGED_GROUPS.get(row['video_id'], 'LEAKAGE_' + row['group_id']),
            'final_split': split_for(row['video_id']),
            'source_frame_path': '../video_dataset_candidate_v1/' + row['frame_path'].replace('\\', '/'),
            'frame_sha256': indexed['frame_sha256'],
        })
    approved_manifest.sort(key=lambda row: (row['video_id'], int(row['frame_number'])))

    video_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    frame_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    hash_videos: dict[str, set[str]] = defaultdict(set)
    for row in approved_manifest:
        video_splits[row['video_id']].add(row['final_split'])
        group_splits[row['split_group_id']].add(row['final_split'])
        frame_splits[row['review_id']].add(row['final_split'])
        hash_splits[row['frame_sha256']].add(row['final_split'])
        hash_videos[row['frame_sha256']].add(row['video_id'])
    duplicate_hashes = {
        digest: sorted(video_ids) for digest, video_ids in sorted(hash_videos.items()) if len(video_ids) > 1
    }
    checks = {
        'no_video_in_multiple_splits': not any(len(value) > 1 for value in video_splits.values()),
        'no_split_group_in_multiple_splits': not any(len(value) > 1 for value in group_splits.values()),
        'no_frame_id_in_multiple_splits': not any(len(value) > 1 for value in frame_splits.values()),
        'no_approved_frame_hash_in_multiple_splits': not any(len(value) > 1 for value in hash_splits.values()),
        'no_rejected_frame_admitted': not ({row['review_id'] for row in rejected} & {row['review_id'] for row in approved_manifest}),
        'no_review_required_frame_admitted': all(row['review_status'] == 'APPROVED' for row in eligible),
        'no_unresolved_label_frame_admitted': not ({row['review_id'] for row in unresolved} & {row['review_id'] for row in approved_manifest}),
        'exactly_270_approved_frames': len(approved_manifest) == 270,
        'exactly_181_venomous_frames': sum(row['verified_class'] == 'Venomous_Snake' for row in approved_manifest) == 181,
        'exactly_89_nonvenomous_frames': sum(row['verified_class'] == 'Non_Venomous_Snake' for row in approved_manifest) == 89,
    }
    if not all(checks.values()):
        raise ValueError(f'Leakage or eligibility check failed: {checks}')

    summary = {
        'candidate_version': 'video_dataset_candidate_v2',
        'source_candidate': '../video_dataset_candidate_v1',
        'authoritative_review_file': '../video_dataset_candidate_v1/review_worklist_manual.csv',
        'eligible_rule': 'label_status == SOURCE_VERIFIED AND review_status == APPROVED',
        'total_approved_frames': 270,
        'excluded_rejected_frames': 11,
        'excluded_unresolved_frames': 327,
        'splits': summarize(approved_manifest, split_manifest),
        'conservative_merged_split_groups': {
            'LEAKAGE_GROUP_STV1747_0001': ['VID_163687650fcdfb63', 'VID_7871d82c6a5700eb'],
            'LEAKAGE_GROUP_STV1608_0010': ['VID_7c9fedb83b820769', 'VID_ffd2ba7c4edda0a1'],
        },
        'leakage_checks': checks,
        'duplicate_approved_hashes_across_source_videos': duplicate_hashes,
        'test_set_policy': ('The independent video test is separate from the frozen image test and must not be used '
                            'for threshold tuning, epoch selection, hyperparameter selection, or model selection.'),
    }

    canonical = json.dumps(
        {'final_split_manifest': split_manifest, 'approved_frame_manifest': approved_manifest, 'dataset_summary': summary},
        ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    )
    fingerprint = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    summary['dataset_fingerprint'] = fingerprint

    temporary = Path(tempfile.mkdtemp(prefix='video_candidate_v2_', dir=ROOT))
    try:
        write_csv(temporary / 'final_split_manifest.csv', split_manifest, list(split_manifest[0]))
        write_csv(temporary / 'approved_frame_manifest.csv', approved_manifest, list(approved_manifest[0]))
        (temporary / 'dataset_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        (temporary / 'dataset_fingerprint.sha256').write_text(fingerprint + '\n', encoding='utf-8')
        if OUTPUT.exists():
            raise FileExistsError(f'Refusing to overwrite frozen candidate: {OUTPUT}')
        os.replace(temporary, OUTPUT)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    if hashlib.sha256(MANUAL.read_bytes()).hexdigest() != manual_hash_before:
        raise RuntimeError('Authoritative manual review CSV changed unexpectedly')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
