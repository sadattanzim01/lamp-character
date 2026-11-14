"""Render docs/TECHNICAL_NOTE.md to docs/TECHNICAL_NOTE.pdf (Letter, 2 pages).
    python docs/render_note.py"""
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

D = Path(__file__).resolve().parent
body = markdown.markdown((D / "TECHNICAL_NOTE.md").read_text(), extensions=["tables"])
css = """
@page { size: Letter; margin: 0.45in 0.5in; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 8.6pt; line-height: 1.32; color: #1a1a1a; }
h1 { font-size: 15pt; margin: 0 0 2px; } h2 { font-size: 10.5pt; margin: 9px 0 3px; color: #2d4f8a; }
p { margin: 3px 0 5px; } ul { margin: 2px 0 4px 16px; padding: 0; } li { margin: 1px 0; }
img { width: 100%; margin: 4px 0 2px; }
table { border-collapse: collapse; width: 100%; margin: 3px 0 5px; font-size: 8pt; }
th, td { border: 1px solid #c9ced6; padding: 2px 5px; text-align: left; vertical-align: top; }
th { background: #eef1f6; } code { font-size: 7.8pt; background: #f2f2f2; padding: 0 2px; border-radius: 2px; }
a { color: #2d4f8a; text-decoration: none; }
"""
html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
(D / "_note.html").write_text(html)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto((D / "_note.html").as_uri()); pg.pdf(path=str(D / "TECHNICAL_NOTE.pdf"), format="Letter",
                                                  margin={"top": "0.45in", "bottom": "0.45in", "left": "0.5in", "right": "0.5in"})
    b.close()
(D / "_note.html").unlink()
print("wrote", D / "TECHNICAL_NOTE.pdf")
