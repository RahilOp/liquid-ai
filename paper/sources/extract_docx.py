#!/usr/bin/env python3
"""Dump a .docx to plain text (headings tagged, tables pipe-delimited).

Used to make the dissertation readable as a source document for the paper.

    python extract_docx.py ../../code-switching/Sharma_Awshesh_Dissertation_Final.docx dissertation_final.txt
"""
import sys

import docx
import docx.oxml.ns as ns
from docx.table import Table
from docx.text.paragraph import Paragraph


def dump(path):
    d = docx.Document(path)
    out = []
    for child in d.element.body.iterchildren():
        if child.tag == ns.qn("w:p"):
            p = Paragraph(child, d)
            text = p.text.strip()
            if not text:
                continue
            style = p.style.name
            out.append(f"[{style}] {text}" if style.startswith("Heading") or style == "Title" else text)
        elif child.tag == ns.qn("w:tbl"):
            out.append("<TABLE>")
            for row in Table(child, d).rows:
                out.append(" | ".join(c.text.strip().replace("\n", " ") for c in row.cells))
            out.append("</TABLE>")
    return "\n".join(out)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    text = dump(src)
    with open(dst, "w") as f:
        f.write(text)
    print(f"{len(text.split())} words -> {dst}")
