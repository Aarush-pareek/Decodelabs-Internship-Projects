"""HTML building blocks: structure and escaping of untrusted text."""

import pytest

from utils.html_components import metric_card, prediction_card, section_header, status_pill


def test_metric_card_escapes_dataset_text():
    html = metric_card("<b>Label</b>", "<script>alert(1)</script>", "a & b")
    assert "<script>" not in html and "<b>Label</b>" not in html
    assert "&lt;script&gt;" in html and "a &amp; b" in html


def test_prediction_card_escapes_class_name_and_clamps_confidence():
    html = prediction_card("<img src=x>", 1.7, "#2a78d6")
    assert "<img" not in html
    assert "100.0%" in html


def test_prediction_card_without_probabilities():
    assert "Not available" in prediction_card("setosa", None, "#2a78d6")


def test_section_header_and_pill():
    assert "Section 3" in section_header(3, "Preprocessing", "sub")
    assert "pill ok" in status_pill("Trained", "ok")
    with pytest.raises(ValueError):
        status_pill("x", "purple")
