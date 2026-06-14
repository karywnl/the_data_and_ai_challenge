#!/usr/bin/env python3
"""Render the pitch deck (deck/slides.md, split on '---') to a landscape PDF.

Run with reportlab available, e.g.:
    uv run --with reportlab python deck/make_deck.py
Produces deck/fitsignal_deck.pdf
"""
from __future__ import annotations

import os
import re

from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "slides.md")
OUT = os.path.join(HERE, "fitsignal_deck.pdf")

PAGE = landscape(A4)
W, H = PAGE
MARGIN = 22 * mm
DARK = (0.11, 0.13, 0.20)
ACCENT = (0.85, 0.20, 0.30)
GREY = (0.35, 0.35, 0.40)


def wrap(c, text, font, size, max_w):
    c.setFont(font, size)
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if c.stringWidth(t, font, size) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_slide(c, lines):
    c.setFillColorRGB(*DARK)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColorRGB(*ACCENT)
    c.rect(0, H - 6 * mm, W, 6 * mm, fill=1, stroke=0)

    x, y = MARGIN, H - MARGIN
    max_w = W - 2 * MARGIN
    in_code = False
    for raw in lines:
        s = raw.rstrip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            # Render diagrams / commands verbatim in a monospace font.
            c.setFillColorRGB(0.7, 0.9, 0.8)
            c.setFont("Courier", 10.5)
            c.drawString(x, y, raw[:118])
            y -= 5.2 * mm
            continue
        if not s:
            y -= 5 * mm
            continue
        if s.startswith("### "):
            c.setFillColorRGB(0.8, 0.85, 0.95)
            for ln in wrap(c, s[4:], "Helvetica-Oblique", 18, max_w):
                c.drawString(x, y, ln); y -= 9 * mm
        elif s.startswith("## "):
            c.setFillColorRGB(1, 1, 1)
            for ln in wrap(c, s[3:], "Helvetica-Bold", 30, max_w):
                c.drawString(x, y, ln); y -= 14 * mm
        elif s.startswith("# "):
            c.setFillColorRGB(1, 1, 1)
            for ln in wrap(c, s[2:], "Helvetica-Bold", 40, max_w):
                c.drawString(x, y, ln); y -= 18 * mm
        elif s.startswith("|"):
            c.setFillColorRGB(0.8, 0.85, 0.92)
            cells = [cell.strip() for cell in s.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            c.setFont("Helvetica", 11)
            cw = max_w / max(len(cells), 1)
            for i, cell in enumerate(cells):
                c.drawString(x + i * cw, y, cell[:38])
            y -= 7 * mm
        elif s.startswith("> "):
            c.setFillColorRGB(*ACCENT)
            c.rect(x, y - 1.5 * mm, 2 * mm, 6 * mm, fill=1, stroke=0)
            c.setFillColorRGB(0.85, 0.88, 0.95)
            for ln in wrap(c, s[2:], "Helvetica-Oblique", 14, max_w - 8 * mm):
                c.drawString(x + 8 * mm, y, ln); y -= 7 * mm
        else:
            txt = re.sub(r"[*`]", "", s)
            bullet = txt.lstrip().startswith(("- ", "1.", "2.", "3."))
            c.setFillColorRGB(0.88, 0.9, 0.95) if not bullet else \
                c.setFillColorRGB(0.78, 0.82, 0.9)
            indent = 6 * mm if bullet else 0
            for ln in wrap(c, txt, "Helvetica", 14, max_w - indent):
                c.drawString(x + indent, y, ln); y -= 7.5 * mm
    c.setFillColorRGB(*GREY)
    c.setFont("Helvetica", 8)
    c.drawRightString(W - MARGIN, 8 * mm, "FitSignal — The Data & AI Challenge")


def main():
    with open(SRC, encoding="utf-8") as f:
        raw = f.read()
    slides = [s.strip("\n") for s in raw.split("\n---\n")]
    c = canvas.Canvas(OUT, pagesize=PAGE)
    for slide in slides:
        draw_slide(c, slide.split("\n"))
        c.showPage()
    c.save()
    print(f"Wrote {OUT} ({len(slides)} slides)")


if __name__ == "__main__":
    main()
