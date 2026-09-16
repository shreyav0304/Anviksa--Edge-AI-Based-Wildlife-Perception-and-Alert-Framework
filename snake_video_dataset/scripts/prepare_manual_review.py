"""Create a video label audit, group proposals, review worklist and contact sheets."""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'video_dataset_candidate_v1'


def read(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write(path: Path, data: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def related_key(name: str) -> tuple[str, str] | None:
    if match := re.match(r'(20\d{6})_(\d{6})', name):
        return 'camera_timestamp', match.group(1) + '_' + match.group(2)[:4]
    if match := re.match(r'(TE1-\d+)--(20\d\d-\d\d-\d\d)--(\d\d-\d\d)', name):
        return 'camera_timestamp', match.group(1) + '_' + match.group(2) + '_' + match.group(3)[:2]
    if match := re.search(r'WhatsApp Video (20\d\d-\d\d-\d\d) at (\d+\.\d+)', name):
        return 'messaging_timestamp', match.group(1) + '_' + match.group(2)
    if match := re.search(r'(STV\d+-\d+)', name, re.I):
        return 'source_asset_code', match.group(1).lower()
    return None


def contact_sheets(videos: list[dict], frames: list[dict]) -> int:
    by_video = defaultdict(list)
    for frame in frames:
        if frame['frame_path']:
            by_video[frame['video_id']].append(frame)
    sheet_root = CANDIDATE / 'contact_sheets'
    sheet_root.mkdir(exist_ok=True)
    pages = 0
    for video in videos:
        candidates = by_video[video['video_id']]
        for start in range(0, len(candidates), 9):
            selected = candidates[start:start + 9]
            canvas = np.full((1220, 1500, 3), 245, dtype=np.uint8)
            header = [f"Video: {video['video_id']}   Page: {start // 9 + 1}",
                      f"File: {video['original_filename']}",
                      f"Source: {video['source_category']}   Label: {video['class_label']}",
                      f"Group: {video['group_id']}   Split: {video['assigned_split']}"]
            for y, line in enumerate(header):
                cv2.putText(canvas, line[:120], (25, 33 + y * 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 2)
            for position, frame in enumerate(selected):
                row, col = divmod(position, 3)
                x, y = 20 + col * 495, 135 + row * 360
                image = cv2.imread(str(CANDIDATE / frame['frame_path']))
                if image is None:
                    raise ValueError(f"Missing candidate frame {frame['frame_path']}")
                h, w = image.shape[:2]
                scale = min(465 / w, 285 / h)
                small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
                canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
                caption = f"{frame['review_id']}  t={frame['timestamp_seconds']}s  frame={frame['frame_number']}"
                cv2.putText(canvas, caption, (x, y + 315), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
            path = sheet_root / f"{video['video_id']}_page_{start // 9 + 1:02d}.jpg"
            if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise OSError(path)
            pages += 1
    return pages


def main() -> None:
    videos = read(CANDIDATE / 'video_manifest.csv')
    frames = read(CANDIDATE / 'review_index.csv')
    by_video = defaultdict(list)
    for frame in frames:
        if frame['frame_path']:
            by_video[frame['video_id']].append(frame)
    audit = []
    for video in videos:
        verified = video['verification_status'].startswith('SOURCE_VERIFIED')
        audit.append(dict(video_id=video['video_id'], original_filename=video['original_filename'], source=video['source_category'],
            group_session_id=video['group_id'], current_snake_class=video['class_label'], species_label=video['species_name'],
            venom_status=video['venom_status'], current_split=video['assigned_split'], candidate_frames=len(by_video[video['video_id']]),
            label_status='SOURCE_VERIFIED' if verified else 'UNKNOWN_OR_UNVERIFIED',
            review_requirement='FRAME_VISIBILITY_AND_QUALITY_REVIEW' if verified else 'USER_CLASS_CONFIRMATION_AND_FRAME_REVIEW'))
    write(CANDIDATE / 'video_label_audit.csv', audit, list(audit[0]))

    clusters = defaultdict(list)
    for video in videos:
        if key := related_key(video['original_filename']):
            clusters[key].append(video)
    proposals = []
    for video in videos:
        candidates = []
        reason = ''
        if key := related_key(video['original_filename']):
            candidates = [v for v in clusters[key] if v['video_id'] != video['video_id'] and v['group_id'] != video['group_id']]
            reason = {'camera_timestamp':'Similar camera/recorder date-time prefix', 'messaging_timestamp':'Same WhatsApp export minute',
                      'source_asset_code':'Same source asset code in filenames'}[key[0]]
        proposals.append(dict(video_id=video['video_id'], filename=video['original_filename'],
            current_group_id=video['group_id'], possible_related_video=';'.join(v['video_id'] for v in candidates),
            reason=reason if candidates else 'No separate relationship supported by filename or metadata',
            recommended_action='ASK_OWNER_TO_CONFIRM_SESSION; KEEP_GROUP_UNCHANGED' if candidates else 'KEEP_GROUP_UNCHANGED'))
    write(CANDIDATE / 'proposed_group_review.csv', proposals, list(proposals[0]))

    worklist = []
    video_by_id = {video['video_id']: video for video in videos}
    for frame in frames:
        if not frame['frame_path']:
            continue
        worklist.append(dict(review_id=f"{frame['video_id']}:{int(frame['frame_number']):08d}", video_id=frame['video_id'],
            frame_number=frame['frame_number'], timestamp_seconds=frame['timestamp_seconds'], frame_path=frame['frame_path'],
            current_label=frame['current_label'], source=frame['source'], group_id=frame['group_id'], split=frame['assigned_split'],
            quality_status=frame['quality_status'], duplicate_status=frame['duplicate_status'], review_status='REVIEW_REQUIRED',
            verified_class=frame['current_label'] if frame['current_label'] in ('Venomous_Snake','Non_Venomous_Snake') else 'UNKNOWN',
            species_name=video_by_id[frame['video_id']]['species_name'] if frame['current_label'] != 'UNKNOWN' else '',
            rejection_reason='', review_notes='', verification_note=''))
    write(CANDIDATE / 'review_worklist.csv', worklist, list(worklist[0]))
    bulk = [dict(video_id=v['video_id'], verified_class='UNKNOWN', species_name='', verification_note='') for v in videos]
    write(CANDIDATE / 'video_label_decisions_template.csv', bulk, list(bulk[0]))
    pages = contact_sheets(videos, worklist)
    print(f'Videos audited: {len(videos)}; candidate frames: {len(worklist)}; contact sheet pages: {pages}; group suggestions: {sum(bool(p["possible_related_video"]) for p in proposals)}')


if __name__ == '__main__':
    main()
