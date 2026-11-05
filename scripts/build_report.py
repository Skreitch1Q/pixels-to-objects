"""Сборка report.pdf из report.md (Markdown → HTML → PDF)."""

from __future__ import annotations

from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent.parent
CSS = """
@page {
  size: A4;
  margin: 18mm 16mm 20mm 16mm;
  @bottom-center {
    content: counter(page);
    font-family: "DejaVu Sans", sans-serif;
    font-size: 8pt;
    color: #666;
  }
}
body {
  font-family: "DejaVu Sans", sans-serif;
  font-size: 9.8pt;
  line-height: 1.45;
  color: #1f2328;
}
h1 {
  font-size: 20pt;
  margin: 0 0 8mm 0;
  color: #16324f;
  border-bottom: 2px solid #16324f;
  padding-bottom: 3mm;
}
h2 {
  font-size: 15pt;
  color: #16324f;
  margin-top: 9mm;
  margin-bottom: 3mm;
  border-bottom: 1px solid #d0d7de;
  padding-bottom: 1.5mm;
}
h3 {
  font-size: 12pt;
  color: #244b6d;
  margin-top: 6mm;
  page-break-after: avoid;
}
p { text-align: justify; }
img {
  display: block;
  max-width: 100%;
  max-height: 205mm;
  margin: 4mm auto;
  page-break-inside: avoid;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 3mm 0 5mm 0;
  font-size: 8.6pt;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #c9d1d9;
  padding: 1.6mm 2mm;
  text-align: right;
}
th { background: #eef3f7; color: #16324f; }
td:first-child, th:first-child { text-align: left; }
code {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8.5pt;
  background: #f3f5f7;
  padding: 0.4mm 1mm;
  border-radius: 1mm;
}
pre {
  background: #f3f5f7;
  border: 1px solid #d0d7de;
  border-radius: 1.5mm;
  padding: 2.5mm;
  white-space: pre-wrap;
  font-size: 8.2pt;
}
pre code { background: none; padding: 0; }
blockquote {
  border-left: 3px solid #8fb3d1;
  margin: 3mm 0;
  padding: 1mm 4mm;
  color: #44515c;
  background: #f7fafc;
}
ul, ol { margin: 2mm 0 3mm 6mm; }
li { margin-bottom: 1mm; }
.caption { font-size: 8.6pt; color: #586069; text-align: center; }
"""


def main() -> None:
    source = ROOT / "report.md"
    body = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    html = (
        "<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    output = ROOT / "report.pdf"
    HTML(string=html, base_url=str(ROOT) + "/").write_pdf(output)
    print(f"PDF сохранён: {output}")


if __name__ == "__main__":
    main()
