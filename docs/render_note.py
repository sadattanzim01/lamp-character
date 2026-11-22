"""Turns the technical note (Markdown) into the 2-page PDF for submission.

    python docs/render_note.py       # reads docs/TECHNICAL_NOTE.md, writes docs/TECHNICAL_NOTE.pdf

What this file does, in plain words:
    1. Converts TECHNICAL_NOTE.md into a web page (HTML), including its tables.
    2. Styles that page with small, tight print settings (fonts, margins,
       table borders) so everything, diagram included, fits on 2 Letter pages,
       the limit the challenge sets.
    3. Opens the page in a hidden Chrome browser (via Playwright) and "prints"
       it to PDF, exactly like Print -> Save as PDF.
    4. Deletes the temporary web page.

Needs: `pip install markdown playwright` (and a Chromium browser for Playwright).
After editing the note, run this again, then check the PDF is still 2 pages.
"""
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

D = Path(__file__).resolve().parent       # the docs/ folder
# Step 1: Markdown -> HTML ("tables" turns the | ... | tables into real HTML tables).
body = markdown.markdown((D / "TECHNICAL_NOTE.md").read_text(), extensions=["tables"])
# Step 2: the print styling. Small fonts and tight spacing are what keep it to 2 pages.
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
# Save it next to the note so the diagram image (architecture.png) is found by its relative path.
(D / "_note.html").write_text(html)
# Step 3: open it in a hidden browser and print it to PDF.
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto((D / "_note.html").as_uri()); pg.pdf(path=str(D / "TECHNICAL_NOTE.pdf"), format="Letter",
                                                  margin={"top": "0.45in", "bottom": "0.45in", "left": "0.5in", "right": "0.5in"})
    b.close()
# Step 4: clean up the temporary page.
(D / "_note.html").unlink()
print("wrote", D / "TECHNICAL_NOTE.pdf")
