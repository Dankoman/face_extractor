#!/usr/bin/env python3
"""GUI reviewer for male faces detected by find_male_faces.py.

Displays each candidate image (sorted by male probability), lets you remove
the file and append the label to remove.txt, or skip to the next image.
Resume support via a state log ensures already processed images are skipped.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk  # type: ignore


Row = Tuple[str, float, float, Path]
StateMap = Dict[str, str]


def load_rows(csv_path: Path, threshold: float) -> List[Row]:
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows: List[Row] = []
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="|")
        for idx, row in enumerate(reader, 1):
            if not row or row[0].lower() == "label":
                continue
            if len(row) < 4:
                print(f"[warn] Skipping malformed row {idx}: {row}")
                continue
            label = row[0]
            try:
                male_val = float(row[1])
            except ValueError:
                male_val = 0.0
            try:
                female_val = float(row[2])
            except ValueError:
                female_val = 0.0
            path = Path(row[3])
            if male_val >= threshold:
                rows.append((label, male_val, female_val, path))

    if not rows:
        raise SystemExit(f"No rows meet threshold {threshold} in {csv_path}")

    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def load_remove_list(remove_file: Path) -> set[str]:
    if not remove_file.exists():
        return set()
    return {line.strip() for line in remove_file.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_to_remove(remove_file: Path, name: str) -> None:
    remove_file.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if remove_file.exists() and remove_file.stat().st_size > 0:
        with remove_file.open("rb") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                needs_newline = True
    with remove_file.open("a", encoding="utf-8") as fh:
        if needs_newline:
            fh.write("\n")
        fh.write(name + "\n")


def load_state(state_file: Path) -> StateMap:
    if not state_file.exists():
        return {}
    mapping: StateMap = {}
    with state_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or "|" not in line:
                continue
            path, action = line.split("|", 1)
            mapping[path] = action
    return mapping


def append_state(handle, path: Path, action: str) -> None:
    handle.write(f"{path}|{action}\n")
    handle.flush()


class ReviewerApp:
    def __init__(
        self,
        rows: List[Row],
        remove_file: Path,
        dry_run: bool,
        state_map: StateMap,
        state_handle,
    ) -> None:
        self.rows = rows
        self.remove_file = remove_file
        self.dry_run = dry_run
        self.state_map = state_map
        self.state_handle = state_handle

        self.existing_names = load_remove_list(remove_file)

        self.root = tk.Tk()
        self.root.title("Male Faces Reviewer")
        self.root.geometry("900x750")
        self.root.configure(bg="#2b2b2b")

        self.info_label = tk.Label(self.root, text="", font=("Arial", 16), fg="#ffffff", bg="#2b2b2b")
        self.info_label.pack(pady=10)

        self.image_label = tk.Label(self.root, bg="#2b2b2b")
        self.image_label.pack(pady=10)

        btn_frame = tk.Frame(self.root, bg="#2b2b2b")
        btn_frame.pack(pady=10)

        self.delete_btn = tk.Button(btn_frame, text="Ta bort", font=("Arial", 14), command=self.delete_current, bg="#d9534f", fg="white", width=12)
        self.delete_btn.pack(side=tk.LEFT, padx=10)

        self.next_btn = tk.Button(btn_frame, text="Nästa", font=("Arial", 14), command=self.skip_current, bg="#5bc0de", fg="white", width=12)
        self.next_btn.pack(side=tk.LEFT, padx=10)

        self.progress_label = tk.Label(self.root, text="", font=("Arial", 12), fg="#cccccc", bg="#2b2b2b")
        self.progress_label.pack(pady=5)

        self.index = -1
        self.photo = None  # to hold ImageTk reference

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.advance()

    def filtered_rows(self) -> List[Row]:
        return [row for row in self.rows if str(row[3]) not in self.state_map]

    def advance(self) -> None:
        while True:
            self.index += 1
            if self.index >= len(self.rows):
                self.finish()
                return
            current_path = str(self.rows[self.index][3])
            if current_path in self.state_map:
                continue
            break
        self.show_current()

    def show_current(self) -> None:
        label, male_prob, female_prob, path = self.rows[self.index]
        self.info_label.config(text=f"{label}  |  male: {male_prob:.3f}  female: {female_prob:.3f}")
        self.progress_label.config(text=f"{self.index + 1} / {len(self.rows)}")

        if not path.exists():
            self.image_label.config(image="", text="Fil saknas", fg="#ffaaaa", bg="#2b2b2b")
            return

        try:
            img = Image.open(path)
            img.thumbnail((800, 600), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo, text="", bg="#2b2b2b")
        except Exception as exc:
            self.image_label.config(image="", text=f"Kunde inte läsa bild: {exc}", fg="#ffaaaa", bg="#2b2b2b")

    def delete_current(self) -> None:
        label, _, _, path = self.rows[self.index]
        path_str = str(path)
        if not path.exists():
            messagebox.showinfo("Info", "Fil saknas – hoppar över.")
            self.record_state(path, "missing")
            self.advance()
            return

        if self.dry_run:
            messagebox.showinfo("Dry-run", "Skulle ta bort filen och appendra i remove.txt")
            self.record_state(path, "deleted")
            self.advance()
            return

        try:
            path.unlink()
        except OSError as exc:
            messagebox.showerror("Fel", f"Kunde inte ta bort filen: {exc}")
            self.record_state(path, f"error:{exc}")
            return

        if label not in self.existing_names:
            append_to_remove(self.remove_file, label)
            self.existing_names.add(label)
        self.record_state(path, "deleted")
        self.advance()

    def skip_current(self) -> None:
        _, _, _, path = self.rows[self.index]
        self.record_state(path, "skip")
        self.advance()

    def record_state(self, path: Path, action: str) -> None:
        self.state_map[str(path)] = action
        if self.state_handle and not self.dry_run:
            append_state(self.state_handle, path, action)

    def finish(self) -> None:
        self.info_label.config(text="Klart!", fg="#00ff8c")
        self.image_label.config(image="", text="", bg="#2b2b2b")
        self.delete_btn.config(state=tk.DISABLED)
        self.next_btn.config(text="Avsluta", command=self.on_close)
        messagebox.showinfo("Klar", "Inga fler poster att granska.")

    def on_close(self) -> None:
        if self.state_handle:
            self.state_handle.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI reviewer for male_faces.csv entries")
    parser.add_argument("csv", type=Path, nargs="?", default=Path("male_faces.csv"),
                        help="CSV export (pipe-delimited) from find_male_faces.py")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Minimum male probability to review (default 0.5)")
    parser.add_argument("--remove-file", type=Path, default=Path("remove.txt"),
                        help="File where labels should be appended")
    parser.add_argument("--state-file", type=Path, default=Path("review_state.log"),
                        help="State log for resume support")
    parser.add_argument("--dry-run", action="store_true", help="Do not delete files or modify remove.txt")
    parser.add_argument("--no-resume", action="store_true", help="Ignore state log (process all rows)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv, args.threshold)
    state_map: StateMap = {}
    state_handle = None
    if not args.no_resume and args.state_file:
        state_map = load_state(args.state_file)
    if not args.dry_run and args.state_file:
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        state_handle = args.state_file.open("a", encoding="utf-8")

    app = ReviewerApp(
        rows=rows,
        remove_file=args.remove_file,
        dry_run=args.dry_run,
        state_map=state_map,
        state_handle=state_handle,
    )
    try:
        app.run()
    finally:
        if state_handle:
            state_handle.close()


if __name__ == "__main__":
    main()

