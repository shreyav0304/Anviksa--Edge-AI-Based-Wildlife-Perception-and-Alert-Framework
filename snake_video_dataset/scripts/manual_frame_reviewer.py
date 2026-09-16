"""Local, human-only reviewer for SOURCE_VERIFIED snake-video candidate frames.

Launch from project root: python snake_video_dataset/scripts/manual_frame_reviewer.py
No model inference, label changes, frame edits, or split changes occur here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'video_dataset_candidate_v1'
SOURCE_WORKLIST = CANDIDATE / 'review_worklist.csv'
MANUAL_WORKLIST = CANDIDATE / 'review_worklist_manual.csv'
AUDIT = CANDIDATE / 'video_label_audit.csv'
VALID_STATUSES = {'APPROVED', 'REJECTED', 'REVIEW_REQUIRED'}
SNAKE_CLASSES = {'Venomous_Snake', 'Non_Venomous_Snake'}
REJECTION_REASONS = ('', 'SNAKE_NOT_VISIBLE', 'UNUSABLE_BLUR', 'CORRUPT_FRAME', 'SCENE_TRANSITION', 'OTHER')


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f'CSV has no header: {path}')
        return list(reader.fieldnames), list(reader)


def atomic_write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=path.parent,
                                         prefix='.review_', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class ReviewStore:
    """CSV persistence and strict verified-video selection; usable headlessly for tests."""

    def __init__(self, source: Path = SOURCE_WORKLIST, manual: Path = MANUAL_WORKLIST,
                 audit: Path = AUDIT, candidate: Path = CANDIDATE, initialize: bool = True):
        self.source = source
        self.manual = manual
        self.candidate = candidate
        self.fields, source_rows = read_csv(source)
        if not {'review_id', 'video_id', 'frame_path', 'review_status', 'verified_class', 'rejection_reason', 'review_notes', 'verification_note'}.issubset(self.fields):
            raise ValueError('Review worklist is missing required columns')
        _, audit_rows = read_csv(audit)
        self.videos = {v['video_id']: v for v in audit_rows if v['label_status'] == 'SOURCE_VERIFIED'}
        if not self.videos or any(v['current_snake_class'] not in SNAKE_CLASSES for v in self.videos.values()):
            raise ValueError('No valid SOURCE_VERIFIED snake videos in the label audit')
        self.verified_ids = [r['review_id'] for r in source_rows if r['video_id'] in self.videos]
        if len(self.verified_ids) != len(set(self.verified_ids)):
            raise ValueError('Duplicate verified review IDs')
        self.source_rows = source_rows
        if manual.exists():
            manual_fields, self.rows = read_csv(manual)
            if manual_fields != self.fields or len(self.rows) != len(source_rows):
                raise ValueError('Manual review file does not match the original worklist schema/length')
            for original, reviewed in zip(source_rows, self.rows):
                for key in ('review_id', 'video_id', 'frame_path', 'frame_number', 'group_id', 'split', 'current_label', 'verified_class'):
                    if original[key] != reviewed[key]:
                        raise ValueError(f'Manual review changed protected metadata at {original["review_id"]}: {key}')
        else:
            self.rows = [dict(row) for row in source_rows]
            if initialize:
                atomic_write(manual, self.fields, self.rows)
        self.by_id = {r['review_id']: r for r in self.rows}
        for review_id in self.verified_ids:
            row = self.by_id[review_id]
            video = self.videos[row['video_id']]
            if row['verified_class'] != video['current_snake_class']:
                raise ValueError(f'Frame class disagrees with source-verified video: {review_id}')
            if row['review_status'] not in VALID_STATUSES or not (candidate / row['frame_path']).is_file():
                raise ValueError(f'Invalid status or missing candidate frame: {review_id}')

    def filtered_ids(self, class_filter: str = 'ALL VERIFIED', video_id: str = 'ALL VIDEOS') -> list[str]:
        return [rid for rid in self.verified_ids if
                (class_filter == 'ALL VERIFIED' or self.by_id[rid]['verified_class'] == class_filter) and
                (video_id == 'ALL VIDEOS' or self.by_id[rid]['video_id'] == video_id)]

    def first_remaining(self, ids: list[str]) -> int:
        return next((i for i, rid in enumerate(ids) if self.by_id[rid]['review_status'] == 'REVIEW_REQUIRED'), 0)

    def counts(self) -> Counter:
        return Counter(self.by_id[rid]['review_status'] for rid in self.verified_ids)

    def decide(self, review_id: str, status: str, rejection_reason: str = '', review_notes: str = '',
               verification_note: str = '') -> None:
        if review_id not in self.verified_ids or status not in VALID_STATUSES:
            raise ValueError('Decision is outside the verified review subset or has invalid status')
        if rejection_reason not in REJECTION_REASONS:
            raise ValueError('Invalid rejection reason')
        row = self.by_id[review_id]
        before = dict(row)
        row['review_status'] = status
        row['rejection_reason'] = rejection_reason if status == 'REJECTED' else ''
        row['review_notes'] = review_notes
        row['verification_note'] = verification_note
        try:
            atomic_write(self.manual, self.fields, self.rows)
        except Exception:
            row.update(before)
            raise


def launch(store: ReviewStore) -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title('Anvīkṣa Snake Video — Manual Frame Reviewer')
    root.geometry('1280x930')
    root.minsize(1000, 760)
    root.configure(bg='#f6f6f2')

    class_filter = tk.StringVar(value='ALL VERIFIED')
    video_filter = tk.StringVar(value='ALL VIDEOS')
    reason = tk.StringVar(value='')
    ids = store.filtered_ids()
    position = store.first_remaining(ids)
    photo_holder = {'image': None}

    top = tk.Frame(root, bg='#f6f6f2')
    top.pack(fill='x', padx=18, pady=10)
    tk.Label(top, text='Manual review of source-verified snake frames', font=('Segoe UI', 18, 'bold'),
             bg='#f6f6f2').pack(anchor='w')
    filters = tk.Frame(top, bg='#f6f6f2')
    filters.pack(fill='x', pady=8)
    tk.Label(filters, text='Class:', bg='#f6f6f2').pack(side='left')
    class_box = ttk.Combobox(filters, textvariable=class_filter, state='readonly', width=22,
                             values=('ALL VERIFIED', 'Venomous_Snake', 'Non_Venomous_Snake'))
    class_box.pack(side='left', padx=8)
    tk.Label(filters, text='Video ID:', bg='#f6f6f2').pack(side='left', padx=(20, 0))
    video_box = ttk.Combobox(filters, textvariable=video_filter, state='readonly', width=27,
                             values=('ALL VIDEOS', *sorted(store.videos)))
    video_box.pack(side='left', padx=8)

    image_label = tk.Label(root, bg='#222222', width=100, height=28)
    image_label.pack(fill='both', expand=True, padx=18, pady=(0, 8))
    detail = tk.Label(root, justify='left', anchor='w', bg='#f6f6f2', font=('Segoe UI', 11), wraplength=1180)
    detail.pack(fill='x', padx=20, pady=4)
    progress = tk.Label(root, bg='#f6f6f2', font=('Segoe UI', 12, 'bold'))
    progress.pack(fill='x', padx=20)

    guidance = ('APPROVE when the snake is visible and the frame provides useful visual information. '
                'Normally retain camouflage, partial visibility, unusual orientation, movement, clutter, '
                'different distances, low light and challenging backgrounds. REJECT only genuinely unusable '
                'frames: snake not visible, corruption, extreme unusable blur, or scene transition.')
    tk.Label(root, text=guidance, justify='left', bg='#e8f2e8', fg='#153b1a', wraplength=1180,
             font=('Segoe UI', 10), padx=10, pady=8).pack(fill='x', padx=18, pady=6)

    notes = tk.Frame(root, bg='#f6f6f2')
    notes.pack(fill='x', padx=18, pady=4)
    tk.Label(notes, text='Rejection reason (optional):', bg='#f6f6f2').grid(row=0, column=0, sticky='w')
    reason_box = ttk.Combobox(notes, textvariable=reason, state='readonly', values=REJECTION_REASONS, width=24)
    reason_box.grid(row=0, column=1, padx=8, sticky='w')
    tk.Label(notes, text='Review notes:', bg='#f6f6f2').grid(row=0, column=2, sticky='w')
    note_entry = tk.Entry(notes, width=55)
    note_entry.grid(row=0, column=3, padx=8, sticky='ew')
    notes.columnconfigure(3, weight=1)
    tk.Label(notes, text='Verification note:', bg='#f6f6f2').grid(row=1, column=0, sticky='w', pady=4)
    verify_entry = tk.Entry(notes, width=80)
    verify_entry.grid(row=1, column=1, columnspan=3, padx=8, sticky='ew')

    controls = tk.Frame(root, bg='#f6f6f2')
    controls.pack(fill='x', padx=18, pady=8)

    def render() -> None:
        if not ids:
            image_label.configure(image='', text='No frames match this filter', fg='white', font=('Segoe UI', 16))
            detail.configure(text='')
            progress.configure(text='')
            return
        rid = ids[position]
        row = store.by_id[rid]
        video = store.videos[row['video_id']]
        try:
            with Image.open(store.candidate / row['frame_path']) as image:
                image = image.convert('RGB')
                image.thumbnail((1160, 545), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
        except Exception as exc:
            messagebox.showerror('Frame could not be opened', str(exc))
            return
        photo_holder['image'] = photo
        image_label.configure(image=photo, text='')
        detail.configure(text=(f"Video ID: {row['video_id']}    File: {video['original_filename']}\n"
                               f"Source: {video['source']}    Group: {row['group_id']}    Species: {video['species_label']}\n"
                               f"Verified class (read-only): {row['verified_class']}    Split: {row['split']}\n"
                               f"Frame: {row['frame_number']}    Timestamp: {row['timestamp_seconds']} s    "
                               f"Review ID: {rid}    Status: {row['review_status']}"))
        count = store.counts()
        overall = store.verified_ids.index(rid) + 1
        progress.configure(text=(f"Frame {overall} / {len(store.verified_ids)}    "
                                 f"Filtered {position + 1} / {len(ids)}    "
                                 f"Approved: {count['APPROVED']}    Rejected: {count['REJECTED']}    "
                                 f"Remaining: {count['REVIEW_REQUIRED']}"))
        reason.set(row['rejection_reason'])
        note_entry.delete(0, 'end')
        note_entry.insert(0, row['review_notes'])
        verify_entry.delete(0, 'end')
        verify_entry.insert(0, row['verification_note'])

    def move(delta: int) -> None:
        nonlocal position
        if ids:
            position = min(max(position + delta, 0), len(ids) - 1)
            render()

    def decide(status: str) -> None:
        if not ids:
            return
        rid = ids[position]
        verification = verify_entry.get().strip()
        if status == 'APPROVED' and not verification:
            verification = 'Human frame review; source metadata class verified'
        try:
            store.decide(rid, status, reason.get() if status == 'REJECTED' else '',
                         note_entry.get(), verification)
        except Exception as exc:
            messagebox.showerror('Decision could not be saved', str(exc))
            return
        render()

    def change_filter(_event=None) -> None:
        nonlocal ids, position
        ids = store.filtered_ids(class_filter.get(), video_filter.get())
        position = store.first_remaining(ids)
        render()

    for title, action, color in (
        ('APPROVE (A)', lambda: decide('APPROVED'), '#3f9950'),
        ('REJECT (R)', lambda: decide('REJECTED'), '#b25048'),
        ('PREVIOUS (←)', lambda: move(-1), '#64778c'),
        ('NEXT (→)', lambda: move(1), '#64778c'),
        ('UNDO (U)', lambda: decide('REVIEW_REQUIRED'), '#777777'),
    ):
        tk.Button(controls, text=title, command=action, bg=color, fg='white',
                  font=('Segoe UI', 13, 'bold'), padx=20, pady=10).pack(side='left', padx=7, expand=True, fill='x')

    class_box.bind('<<ComboboxSelected>>', change_filter)
    video_box.bind('<<ComboboxSelected>>', change_filter)

    def shortcut(event) -> None:
        key = event.keysym.lower()
        if key == 'left':
            move(-1)
            return
        if key == 'right':
            move(1)
            return
        if isinstance(root.focus_get(), (tk.Entry, ttk.Combobox)):
            return
        if key == 'a':
            decide('APPROVED')
        elif key == 'r':
            decide('REJECTED')
        elif key == 'u':
            decide('REVIEW_REQUIRED')

    root.bind('<Key>', shortcut)
    render()
    root.mainloop()


def self_test() -> None:
    """Exercise loading and save/resume in a temporary directory only."""
    original_hash = hashlib.sha256(SOURCE_WORKLIST.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='snake_reviewer_test_') as directory:
        temporary = Path(directory) / 'review_worklist_manual_test.csv'
        store = ReviewStore(manual=temporary)
        assert len(store.verified_ids) == 281, len(store.verified_ids)
        classes = Counter(store.by_id[rid]['verified_class'] for rid in store.verified_ids)
        assert classes == {'Venomous_Snake': 191, 'Non_Venomous_Snake': 90}, classes
        for rid in store.verified_ids:
            with Image.open(store.candidate / store.by_id[rid]['frame_path']) as image:
                image.verify()
        first, second = store.verified_ids[:2]
        store.decide(first, 'APPROVED', verification_note='Temporary test; not a real review')
        resumed = ReviewStore(manual=temporary)
        assert resumed.by_id[first]['review_status'] == 'APPROVED'
        resumed.decide(second, 'REJECTED', 'SNAKE_NOT_VISIBLE', 'Temporary test')
        resumed = ReviewStore(manual=temporary)
        assert resumed.by_id[second]['review_status'] == 'REJECTED'
        resumed.decide(first, 'REVIEW_REQUIRED')
        resumed.decide(second, 'REVIEW_REQUIRED')
        resumed = ReviewStore(manual=temporary)
        assert resumed.by_id[first]['review_status'] == 'REVIEW_REQUIRED'
        assert resumed.by_id[second]['review_status'] == 'REVIEW_REQUIRED'
    assert hashlib.sha256(SOURCE_WORKLIST.read_bytes()).hexdigest() == original_hash
    print('PASS: 281 images load; 191 venomous, 90 nonvenomous; APPROVE, REJECT, resume and UNDO persist in temporary CSV; original worklist unchanged; temporary files removed.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manual-file', type=Path, default=MANUAL_WORKLIST,
                        help='Separate resumable decision CSV; original review_worklist.csv is read-only')
    parser.add_argument('--self-test', action='store_true', help='Headless loading and temporary save/resume validation')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    store = ReviewStore(manual=args.manual_file)
    launch(store)


if __name__ == '__main__':
    main()
