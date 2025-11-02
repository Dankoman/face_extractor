#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Iterable, Tuple


Row = Tuple[str, float, float, str]


HTML_TEMPLATE_HEAD = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 1.5rem; background: #f7f7f7; }}
    h1 {{ margin-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 2px 6px rgba(0,0,0,.1); }}
    th, td {{ padding: 0.6rem; border-bottom: 1px solid #ddd; text-align: left; }}
    th {{ cursor: pointer; background: #fafafa; position: sticky; top: 0; }}
    tr:hover {{ background: #f1f1f1; }}
    img.thumb {{ max-height: 160px; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,.2); }}
    .controls {{ margin-bottom: 1rem; }}
  </style>
  <script>
    function sortTable(n, numeric) {{
      const table = document.getElementById('faces-table');
      let switching = true;
      let dir = 'desc';
      while (switching) {{
        switching = false;
        const rows = table.rows;
        for (let i = 1; i < rows.length - 1; i++) {{
          let shouldSwitch = false;
          let x = rows[i].getElementsByTagName('TD')[n];
          let y = rows[i + 1].getElementsByTagName('TD')[n];
          let xVal = x.getAttribute('data-sort') || x.innerText;
          let yVal = y.getAttribute('data-sort') || y.innerText;
          if (numeric) {{
            xVal = parseFloat(xVal) || 0;
            yVal = parseFloat(yVal) || 0;
          }} else {{
            xVal = xVal.toLowerCase();
            yVal = yVal.toLowerCase();
          }}
          if (dir === 'asc' ? xVal > yVal : xVal < yVal) {{
            shouldSwitch = true;
            break;
          }}
        }}
        if (shouldSwitch) {{
          rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
          switching = true;
        }} else if (dir === 'desc' && !switching) {{
          dir = 'asc';
          switching = true;
        }}
      }}
    }}
  </script>
</head>
<body>
  <h1>{title}</h1>
  <div class=\"controls\">Click column headers to sort. Thumbnails load from local file paths.</div>
  <table id=\"faces-table\">
    <thead>
      <tr>
        <th onclick=\"sortTable(0, false)\">Label</th>
        <th onclick=\"sortTable(1, true)\">Male prob</th>
        <th onclick=\"sortTable(2, true)\">Female prob</th>
        <th>Thumbnail</th>
        <th onclick=\"sortTable(4, false)\">Image path</th>
      </tr>
    </thead>
    <tbody>
"""

HTML_TEMPLATE_FOOT = """    </tbody>
  </table>
</body>
</html>
"""


def generate_html(rows: Iterable[Row], output: Path, title: str = "Male Faces Report") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        fh.write(HTML_TEMPLATE_HEAD.format(title=html.escape(title)))
        for label, male_prob, female_prob, path in rows:
            label_htm = html.escape(label)
            male_str = f"{male_prob:.3f}"
            female_str = f"{female_prob:.3f}"
            path_htm = html.escape(path)
            img_src = "file://" + path_htm
            fh.write(
                "      <tr>\n"
                f"        <td>{label_htm}</td>\n"
                f"        <td data-sort=\"{male_prob}\">{male_str}</td>\n"
                f"        <td data-sort=\"{female_prob}\">{female_str}</td>\n"
                f"        <td><img class=\"thumb\" src=\"{img_src}\" loading=\"lazy\" alt=\"{label_htm}\"></td>\n"
                f"        <td>{path_htm}</td>\n"
                "      </tr>\n"
            )
        fh.write(HTML_TEMPLATE_FOOT)


def load_rows(csv_path: Path, min_prob: float) -> list[Row]:
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    rows: list[Row] = []
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="|")
        for row in reader:
            if not row:
                continue
            if row[0].lower() == "label":
                continue
            if len(row) < 4:
                continue
            label = row[0]
            try:
                male = float(row[1])
            except ValueError:
                male = 0.0
            try:
                female = float(row[2])
            except ValueError:
                female = 0.0
            path = row[3]
            if male >= min_prob:
                rows.append((label, male, female, path))
    if not rows:
        raise SystemExit("No data rows found in CSV.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an HTML report from male_faces.csv")
    parser.add_argument("csv", type=Path, nargs="?", default=Path("male_faces.csv"),
                        help="Input CSV (pipe-delimited) with header label|male_prob|female_prob|path")
    parser.add_argument("--output", type=Path, default=Path("male_faces.html"),
                        help="Output HTML file")
    parser.add_argument("--title", default="Male Faces Report", help="Title for the HTML page")
    parser.add_argument("--min-prob", type=float, default=0.5,
                        help="Minimum male probability to include (default 0.5)")
    args = parser.parse_args()

    rows = load_rows(args.csv, args.min_prob)
    rows.sort(key=lambda r: r[1], reverse=True)
    generate_html(rows, args.output, args.title)
    print(f"Generated {args.output}")


if __name__ == "__main__":
    main()
