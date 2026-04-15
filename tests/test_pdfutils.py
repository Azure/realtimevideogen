#!/usr/bin/env python3

from pdf_utils import parse_pdf


def test_parse_pdf() -> None:
    pdf_path = "tests/data/blank.pdf"
    text, images = parse_pdf(pdf_path)

    assert text is not None
    assert text == [""]

    assert images is not None
    assert len(images) == 1
    assert images[0].startswith("data:image/jpeg;base64,")
