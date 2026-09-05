#!/usr/bin/env python3
"""Offline Tkinter reviewer for ANVIKSA's 324 unresolved snake images.

This tool records human decisions only. It never changes source images,
datasets, models, splits, or contact sheets. CSV writes are atomic.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


APP_TITLE = "ANVIKSA Snake Dataset Manual Review"
NOT_REVIEWED = "NOT REVIEWED"
BASE_FIELDS = [
    "review_id", "image_path", "species", "parent_class", "review_type",
    "suspected_match_path", "suspected_match_split", "frozen_test_involved",
    "similarity_summary", "contact_sheet_path", "human_decision", "human_notes",
    "final_recommended_status",
]
ADDITIONAL_FIELDS = ["reviewed_at", "review_status"]
STANDARD_VALID = {
    "DUPLICATE_REVIEW": ["KEEP", "DUPLICATE", "UNSURE"],
    "FROZEN_TEST_REVIEW": ["KEEP", "TEST LEAKAGE", "UNSURE"],
    "LABEL_REVIEW": ["LABEL CORRECT", "LABEL INCORRECT", "UNSURE"],
}
FILTERS = [
    "ALL", "UNREVIEWED", "REVIEWED", "DUPLICATE REVIEW",
    "FROZEN TEST REVIEW", "LABEL REVIEW", "UNSURE",
]
COUNTER_LABELS = [
    "TOTAL CASES", "REVIEWED", "UNREVIEWED", "KEEP", "DUPLICATE",
    "TEST LEAKAGE", "LABEL CORRECT", "LABEL INCORRECT", "UNSURE",
]


class DecisionError(ValueError):
    pass


class DecisionStore:
    def __init__(self, csv_path: Path, review_root: Path) -> None:
        self.csv_path = csv_path.resolve()
        self.review_root = review_root.resolve()
        self.rows: list[dict[str, str]] = []
        self.fieldnames: list[str] = []
        self.by_id: dict[str, dict[str, str]] = {}
        self.load()

    def load(self) -> None:
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Decision CSV not found: {self.csv_path}")
        with self.csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            self.fieldnames = list(reader.fieldnames or [])
            self.rows = list(reader)
        missing = [field for field in BASE_FIELDS if field not in self.fieldnames]
        if missing:
            raise DecisionError(f"Decision CSV is missing required columns: {missing}")
        ids = [row["review_id"] for row in self.rows]
        if len(ids) != len(set(ids)):
            raise DecisionError("Duplicate review IDs found")
        if not ids or any(not review_id for review_id in ids):
            raise DecisionError("Review IDs must be populated")
        for field in ADDITIONAL_FIELDS:
            if field not in self.fieldnames:
                self.fieldnames.append(field)
            for row in self.rows:
                row.setdefault(field, "")
        self.by_id = {row["review_id"]: row for row in self.rows}

    def contact_sheet(self, row: dict[str, str]) -> Path:
        return (self.review_root / row["contact_sheet_path"]).resolve()

    def valid_decisions(self, row: dict[str, str]) -> list[str]:
        valid = list(STANDARD_VALID.get(row["review_type"], ["KEEP", "UNSURE"]))
        # Label cases in this queue originate from duplicate metadata conflicts.
        if row["review_type"] == "LABEL_REVIEW" and "DUPLICATE" not in valid:
            valid.insert(-1, "DUPLICATE")
        if row.get("frozen_test_involved", "").upper() == "YES" and "TEST LEAKAGE" not in valid:
            valid.insert(-1, "TEST LEAKAGE")
        return valid

    def validate(self, review_id: str, decision: str) -> dict[str, str]:
        if review_id not in self.by_id:
            raise DecisionError(f"Unknown review ID: {review_id}")
        row = self.by_id[review_id]
        if not Path(row["image_path"]).is_file():
            raise DecisionError(f"Source row image is missing: {row['image_path']}")
        if not self.contact_sheet(row).is_file():
            raise DecisionError("CONTACT SHEET MISSING; case must remain unresolved")
        if decision not in self.valid_decisions(row):
            raise DecisionError(f"{decision!r} is invalid for {row['review_type']}")
        return row

    def set_decision(self, review_id: str, decision: str, notes: str) -> None:
        row = self.validate(review_id, decision)
        row["human_decision"] = decision
        row["human_notes"] = notes.strip()
        row["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        row["review_status"] = "REVIEWED"
        # Deliberately do not infer or pre-fill a downstream status.
        self.atomic_save()

    def atomic_save(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", newline="", encoding="utf-8-sig", delete=False,
                dir=self.csv_path.parent, prefix=self.csv_path.name + ".", suffix=".tmp",
            ) as handle:
                temp_path = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.csv_path)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

    def counts(self) -> dict[str, int]:
        decisions = Counter(row.get("human_decision", "") for row in self.rows)
        reviewed = sum(bool(row.get("human_decision", "")) for row in self.rows)
        return {
            "TOTAL CASES": len(self.rows), "REVIEWED": reviewed,
            "UNREVIEWED": len(self.rows) - reviewed,
            "KEEP": decisions["KEEP"], "DUPLICATE": decisions["DUPLICATE"],
            "TEST LEAKAGE": decisions["TEST LEAKAGE"],
            "LABEL CORRECT": decisions["LABEL CORRECT"],
            "LABEL INCORRECT": decisions["LABEL INCORRECT"],
            "UNSURE": decisions["UNSURE"],
        }

    def category_counts(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for category in sorted({row["review_type"] for row in self.rows}):
            rows = [row for row in self.rows if row["review_type"] == category]
            decisions = Counter(row["human_decision"] or NOT_REVIEWED for row in rows)
            result[category] = {"TOTAL": len(rows), **dict(sorted(decisions.items()))}
        return result

    def write_status(self, path: Path) -> None:
        counts = self.counts()
        lines = ["# ANVIKSA Interactive Manual Review Status", ""]
        lines += [f"- {label}: {counts[label]}" for label in COUNTER_LABELS]
        lines += ["", "## Decisions by review category", ""]
        for category, values in self.category_counts().items():
            lines.append(f"### {category}")
            lines.append("")
            lines += [f"- {key}: {value}" for key, value in values.items()]
            lines.append("")
        lines += [
            "## Safety status", "",
            "- TRAINING PERFORMED: NO", "- DATASET INTEGRATION PERFORMED: NO",
            "- SOURCE IMAGES MODIFIED: NO", "- FROZEN TEST SET MODIFIED: NO",
            "- MODELS MODIFIED: NO", "- VIDEO DATA MODIFIED: NO",
            "- ANVIKSA MAIN UI MODIFIED: NO", "",
            "Technical approval does not establish copyright or training-reuse rights. Dataset rights clearance remains a separate requirement.",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ReviewSession:
    def __init__(self, store: DecisionStore) -> None:
        self.store = store
        self.filter_name = "ALL"
        self.visible_ids: list[str] = []
        self.index = 0
        self.apply_filter("ALL")

    def matches(self, row: dict[str, str], filter_name: str) -> bool:
        decision = row.get("human_decision", "")
        return {
            "ALL": True,
            "UNREVIEWED": not decision,
            "REVIEWED": bool(decision),
            "DUPLICATE REVIEW": row["review_type"] == "DUPLICATE_REVIEW",
            "FROZEN TEST REVIEW": row["review_type"] == "FROZEN_TEST_REVIEW",
            "LABEL REVIEW": row["review_type"] == "LABEL_REVIEW",
            "UNSURE": decision == "UNSURE",
        }[filter_name]

    def apply_filter(self, filter_name: str, keep_id: str | None = None) -> None:
        if filter_name not in FILTERS:
            raise DecisionError(f"Unknown filter: {filter_name}")
        self.filter_name = filter_name
        self.visible_ids = [row["review_id"] for row in self.store.rows if self.matches(row, filter_name)]
        if keep_id in self.visible_ids:
            self.index = self.visible_ids.index(keep_id)
        else:
            self.index = min(self.index, max(0, len(self.visible_ids) - 1))

    def current(self) -> dict[str, str] | None:
        return self.store.by_id[self.visible_ids[self.index]] if self.visible_ids else None

    def previous(self) -> None:
        if self.visible_ids:
            self.index = max(0, self.index - 1)

    def next(self) -> None:
        if self.visible_ids:
            self.index = min(len(self.visible_ids) - 1, self.index + 1)

    def first_unreviewed(self) -> bool:
        for row in self.store.rows:
            if not row.get("human_decision", ""):
                if row["review_id"] not in self.visible_ids:
                    self.apply_filter("UNREVIEWED")
                self.index = self.visible_ids.index(row["review_id"])
                return True
        return False


class ReviewApp:
    def __init__(self, root: tk.Tk, store: DecisionStore, status_path: Path) -> None:
        self.root = root
        self.store = store
        self.status_path = status_path
        self.session = ReviewSession(store)
        self.session.first_unreviewed()
        self.photo: ImageTk.PhotoImage | None = None
        self.resize_after: str | None = None
        self.root.title(APP_TITLE)
        self.root.geometry("1280x900")
        self.root.minsize(900, 650)
        self.build()
        self.bind_keys()
        self.render()

    def build(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        header = ttk.Frame(self.root, padding=10)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text=APP_TITLE, font=("Segoe UI", 17, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")
        self.case_var = tk.StringVar(); self.progress_var = tk.StringVar(); self.type_var = tk.StringVar()
        self.risk_var = tk.StringVar(); self.decision_var = tk.StringVar()
        ttk.Label(header, textvariable=self.case_var).grid(row=1, column=0, sticky="w", padx=(0, 20))
        ttk.Label(header, textvariable=self.progress_var).grid(row=1, column=1, sticky="w")
        ttk.Label(header, textvariable=self.type_var).grid(row=2, column=0, sticky="w", padx=(0, 20))
        ttk.Label(header, textvariable=self.risk_var).grid(row=2, column=1, sticky="w")
        ttk.Label(header, textvariable=self.decision_var, font=("Segoe UI", 10, "bold")).grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(header, text="Filter:").grid(row=1, column=2, sticky="e")
        self.filter_var = tk.StringVar(value=self.session.filter_name)
        combo = ttk.Combobox(header, textvariable=self.filter_var, values=FILTERS, state="readonly", width=24)
        combo.grid(row=1, column=3, sticky="e"); combo.bind("<<ComboboxSelected>>", self.on_filter)

        stats = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        stats.grid(row=1, column=0, sticky="ew")
        self.stats_var = tk.StringVar()
        ttk.Label(stats, textvariable=self.stats_var).pack(side="left")

        image_frame = ttk.Frame(self.root, padding=(10, 0))
        image_frame.grid(row=2, column=0, sticky="nsew")
        image_frame.columnconfigure(0, weight=1); image_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(image_frame, bg="#202020", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.schedule_image)

        decisions = ttk.LabelFrame(self.root, text="Decision", padding=8)
        decisions.grid(row=3, column=0, sticky="ew", padx=10, pady=8)
        self.decision_buttons = ttk.Frame(decisions); self.decision_buttons.pack(fill="x")
        notes_frame = ttk.Frame(decisions); notes_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(notes_frame, text="Reviewer Notes:").pack(anchor="w")
        self.notes = tk.Text(notes_frame, height=3, wrap="word")
        self.notes.pack(fill="x")

        navigation = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        navigation.grid(row=4, column=0, sticky="ew")
        ttk.Button(navigation, text="Previous", command=self.previous).pack(side="left")
        ttk.Button(navigation, text="Next", command=self.next).pack(side="left", padx=6)
        ttk.Button(navigation, text="Resume First Unreviewed", command=self.first_unreviewed).pack(side="left")
        ttk.Button(navigation, text="Open Full Size", command=self.open_full_size).pack(side="left", padx=6)
        ttk.Button(navigation, text="Refresh CSV", command=self.reload).pack(side="left")
        notice = "Technical approval does not establish copyright or training-reuse rights. Dataset rights clearance remains a separate requirement."
        ttk.Label(navigation, text=notice, foreground="#a12820", wraplength=570, justify="right").pack(side="right")

    def bind_keys(self) -> None:
        self.root.bind("<Left>", lambda _e: self.previous())
        self.root.bind("<Right>", lambda _e: self.next())
        for key, decision in [("k", "KEEP"), ("d", "DUPLICATE"), ("t", "TEST LEAKAGE"), ("c", "LABEL CORRECT"), ("i", "LABEL INCORRECT"), ("u", "UNSURE")]:
            self.root.bind(f"<{key}>", lambda event, value=decision: self.shortcut(event, value))

    def shortcut(self, event: tk.Event, decision: str) -> None:
        if event.widget == self.notes:
            return
        row = self.session.current()
        if row and decision in self.store.valid_decisions(row):
            self.save_decision(decision)

    def on_filter(self, _event: tk.Event | None = None) -> None:
        current = self.session.current()
        self.session.apply_filter(self.filter_var.get(), current["review_id"] if current else None)
        self.render()

    def previous(self) -> None:
        self.session.previous(); self.render()

    def next(self) -> None:
        self.session.next(); self.render()

    def first_unreviewed(self) -> None:
        if self.session.first_unreviewed():
            self.filter_var.set(self.session.filter_name); self.render()
        else:
            messagebox.showinfo(APP_TITLE, "All review cases currently have decisions.")

    def reload(self) -> None:
        current = self.session.current()
        self.store.load()
        self.session.apply_filter(self.session.filter_name, current["review_id"] if current else None)
        self.render()

    def save_decision(self, decision: str) -> None:
        row = self.session.current()
        if not row:
            return
        try:
            self.store.set_decision(row["review_id"], decision, self.notes.get("1.0", "end-1c"))
            self.store.write_status(self.status_path)
        except (DecisionError, OSError) as exc:
            messagebox.showerror("Decision not saved", str(exc)); return
        current_id = row["review_id"]
        self.session.apply_filter(self.session.filter_name, current_id)
        if self.session.filter_name == "UNREVIEWED":
            self.session.index = min(self.session.index, max(0, len(self.session.visible_ids) - 1))
        self.render()

    def render(self) -> None:
        row = self.session.current()
        counts = self.store.counts()
        self.stats_var.set("   |   ".join(f"{label}: {counts[label]}" for label in COUNTER_LABELS))
        for child in self.decision_buttons.winfo_children():
            child.destroy()
        self.notes.delete("1.0", "end")
        if not row:
            self.case_var.set("Review Case: NONE"); self.progress_var.set("Progress: 0 / 0")
            self.type_var.set("Review Type: —"); self.risk_var.set("Frozen Test Risk: —")
            self.decision_var.set("Current Decision: —")
            self.canvas.delete("all"); self.canvas.create_text(20, 20, anchor="nw", text="No cases match this filter.", fill="white")
            return
        self.case_var.set(f"Review Case: {row['review_id']}")
        self.progress_var.set(f"Progress: {self.session.index + 1} / {len(self.session.visible_ids)} (queue total {len(self.store.rows)})")
        self.type_var.set(f"Review Type: {row['review_type']}")
        self.risk_var.set(f"Frozen Test Risk: {row['frozen_test_involved']}")
        self.decision_var.set(f"Current Decision: {row['human_decision'] or NOT_REVIEWED}")
        self.notes.insert("1.0", row.get("human_notes", ""))
        for decision in self.store.valid_decisions(row):
            ttk.Button(self.decision_buttons, text=decision, command=lambda value=decision: self.save_decision(value)).pack(side="left", padx=(0, 7))
        self.render_image()

    def schedule_image(self, _event: tk.Event) -> None:
        if self.resize_after:
            self.root.after_cancel(self.resize_after)
        self.resize_after = self.root.after(100, self.render_image)

    def render_image(self) -> None:
        self.resize_after = None
        row = self.session.current()
        self.canvas.delete("all")
        if not row:
            return
        path = self.store.contact_sheet(row)
        if not path.is_file():
            self.photo = None
            self.canvas.create_text(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, text="CONTACT SHEET MISSING\nCase remains unresolved.", fill="#ff7777", font=("Segoe UI", 18, "bold"), justify="center")
            return
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                area = (max(100, self.canvas.winfo_width() - 20), max(100, self.canvas.winfo_height() - 20))
                image.thumbnail(area, Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, image=self.photo, anchor="center")
        except OSError as exc:
            self.photo = None
            self.canvas.create_text(20, 20, anchor="nw", text=f"CONTACT SHEET UNREADABLE\n{exc}", fill="#ff7777")

    def open_full_size(self) -> None:
        row = self.session.current()
        if not row:
            return
        path = self.store.contact_sheet(row)
        if not path.is_file():
            messagebox.showwarning(APP_TITLE, "CONTACT SHEET MISSING"); return
        top = tk.Toplevel(self.root); top.title(f"{row['review_id']} — Full Size")
        top.geometry("1200x800")
        frame = ttk.Frame(top); frame.pack(fill="both", expand=True)
        canvas = tk.Canvas(frame, bg="#202020"); xbar = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview); ybar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        canvas.grid(row=0, column=0, sticky="nsew"); ybar.grid(row=0, column=1, sticky="ns"); xbar.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1); frame.columnconfigure(0, weight=1)
        with Image.open(path) as source:
            photo = ImageTk.PhotoImage(source.convert("RGB"))
        canvas.create_image(0, 0, image=photo, anchor="nw"); canvas.configure(scrollregion=(0, 0, photo.width(), photo.height()))
        canvas.image = photo


def self_test(real_csv: Path, review_root: Path) -> None:
    """Functional test using an isolated CSV; no real decision is written."""
    with tempfile.TemporaryDirectory(prefix="anviksa_manual_review_test_") as temp_name:
        temp = Path(temp_name)
        test_csv = temp / "manual_decisions.csv"
        shutil.copyfile(real_csv, test_csv)
        store = DecisionStore(test_csv, review_root)
        assert len(store.rows) == 324
        assert all(store.contact_sheet(row).is_file() for row in store.rows)
        session = ReviewSession(store)
        first = session.current()["review_id"]
        session.next(); assert session.current()["review_id"] != first
        session.previous(); assert session.current()["review_id"] == first
        for filter_name in FILTERS:
            session.apply_filter(filter_name)
        session.apply_filter("ALL"); assert session.first_unreviewed()
        duplicate = next(row for row in store.rows if row["review_type"] == "DUPLICATE_REVIEW")
        try:
            store.validate(duplicate["review_id"], "LABEL CORRECT")
            raise AssertionError("Invalid decision was accepted")
        except DecisionError:
            pass
        store.set_decision(duplicate["review_id"], "KEEP", "temporary persistence test")
        reopened = DecisionStore(test_csv, review_root)
        assert reopened.by_id[duplicate["review_id"]]["human_decision"] == "KEEP"
        assert reopened.by_id[duplicate["review_id"]]["human_notes"] == "temporary persistence test"
        reopened.by_id[duplicate["review_id"]]["contact_sheet_path"] = "missing.png"
        try:
            reopened.validate(duplicate["review_id"], "KEEP")
            raise AssertionError("Missing contact sheet was accepted")
        except DecisionError:
            pass
    print("SELF-TEST PASS: navigation, filters, validation, atomic autosave, notes, resume data, reopen persistence, and missing-sheet handling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--review-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Create/update/destroy the UI without entering the main loop")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    review_root = (args.review_root or project_root / "results/new_snake_image_manual_review").resolve()
    csv_path = (args.csv or review_root / "manual_decisions.csv").resolve()
    if args.self_test:
        self_test(csv_path, review_root); return 0
    store = DecisionStore(csv_path, review_root)
    status_path = review_root / "interactive_review_status.md"
    store.write_status(status_path)
    root = tk.Tk()
    app = ReviewApp(root, store, status_path)
    if args.smoke_test:
        root.update_idletasks(); root.update(); root.destroy()
        print(f"SMOKE TEST PASS: loaded {len(store.rows)} cases and rendered {app.session.current()['review_id']}")
        return 0
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
