#!/usr/bin/env python3
"""
Identifiera mappar som innehåller bilder på flera olika personer (impurities).
Baseras på 'low_confidence_images.csv' från train_confidence.py.

Ger en rapport om vilka mappar som har flest felaktiga bilder och vem de liknar.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

def main() -> None:
    parser = argparse.ArgumentParser(description="Hitta mappar med flera personer.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("low_confidence_images.csv"),
        help="Sökväg till CSV-filen (default: low_confidence_images.csv)",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.4,
        help="Ignorera mismatches där top_confidence är lägre än detta (default: 0.4)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Filen {args.csv} saknas.")
        return

    # Data structure: folder -> total_images, mismatches -> list of (top_label, top_conf)
    folder_stats: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "mismatches": []})

    print(f"Läser in {args.csv}...")
    with args.csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row.get("folder_name", "").strip()
            label = row.get("label", "").strip() 
            # Fallback if folder_name is missing (old CSV format compat)
            if not folder and row.get("path"):
                 folder = Path(row["path"]).parent.name
            
            if not folder:
                continue

            folder_stats[folder]["total"] += 1

            mismatch = row.get("mismatch", "").lower() == "yes"
            top_label = row.get("top_label", "").strip()
            # Try parsing confidence, default to 0.0 if failed
            try:
                top_conf = float(row.get("top_confidence", 0.0))
            except ValueError:
                top_conf = 0.0

            if mismatch and top_conf >= args.min_conf:
                folder_stats[folder]["mismatches"].append((top_label, top_conf))

    # Filter folders that have at least one valid mismatch
    impure_folders = []
    for folder, data in folder_stats.items():
        if data["mismatches"]:
            impure_folders.append((folder, data))

    # Sort by number of mismatches (descending)
    impure_folders.sort(key=lambda x: len(x[1]["mismatches"]), reverse=True)

    print(f"\nHittade {len(impure_folders)} mappar med potentiella fel (minst {args.min_conf} confidence).\n")
    print("-" * 60)

    for folder, data in impure_folders:
        count = len(data["mismatches"])
        total = data["total"]
        pct = (count / total) * 100 if total > 0 else 0
        
        print(f"{folder} ({total} bilder):")
        print(f"  ⚠️  {count} bilder verkar felaktiga ({pct:.1f}%)")

        # Group mismatches by label
        grouped_errors: Dict[str, List[float]] = defaultdict(list)
        for label, conf in data["mismatches"]:
            grouped_errors[label].append(conf)

        # Sort error groups by count
        sorted_errors = sorted(grouped_errors.items(), key=lambda x: len(x[1]), reverse=True)

        for label, confs in sorted_errors[:5]: # Show top 5 impurities
            avg_conf = sum(confs) / len(confs)
            print(f"      -> {len(confs)} st liknar '{label}' (snitt-conf: {avg_conf:.2f})")
        
        if len(sorted_errors) > 5:
            print(f"      -> ... och {len(sorted_errors) - 5} andra labels.")
        
        print("")

if __name__ == "__main__":
    main()
