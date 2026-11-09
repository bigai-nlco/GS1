"""Convert the supplied Pareto PDF to self-contained vector SVGs and a PNG preview.

uv run --no-project --with pymupdf python scripts/convert_pareto.py
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def convert(source, directory):
    import pymupdf

    source, directory = Path(source), Path(directory)
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    with pymupdf.open(source) as document:
        if len(document) != 1:
            raise ValueError("Expected the supplied single-page GS1 Pareto figure")
        page = document[0]
        svg = ET.fromstring(page.get_svg_image(text_as_path=True))
        svg.set("role", "img")
        svg.set("aria-labelledby", "title desc")
        title = ET.Element(f"{{{namespace}}}title", id="title")
        title.text = "GS1 parameter–performance Pareto frontier"
        desc = ET.Element(f"{{{namespace}}}desc", id="desc")
        desc.text = (
            "Technical-report snapshot, 6 November 2026: Decision Index 0.4.1 public text scores "
            "for GS1 0.8B, 2B, 4B and 9B are 56.15, 62.22, 68.95 and 70.47. "
            "The frontier refers to the 33 displayed comparison models. "
            "Jev 1.14 has a reference score of 60.70; its parameter count is undisclosed."
        )
        metadata = ET.Element(f"{{{namespace}}}metadata")
        metadata.text = json.dumps(
            {"source": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        )
        svg.insert(0, metadata)
        svg.insert(0, desc)
        svg.insert(0, title)
        content = ET.tostring(svg, encoding="utf-8", xml_declaration=True)
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("gs1_pareto_frontier.svg", "model-quality.svg"):
            (directory / name).write_bytes(content)
        page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(directory / "model-quality.png")
    print(f"Converted {source.name} to vector SVG; all plotted coordinates and scores preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("assets/gs1_pareto_frontier.pdf"))
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    args = parser.parse_args()
    convert(args.source, args.assets)


if __name__ == "__main__":
    main()
