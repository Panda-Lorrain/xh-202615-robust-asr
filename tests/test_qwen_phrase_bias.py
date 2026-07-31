import io
import os
import sys
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from qwen_phrase_bias import AcousticTopKPhraseBias, load_phrases
from exp_qwen_phrase_bias import select_stratified


def test_phrase_start_is_boosted_only_inside_top_k():
    proc = AcousticTopKPhraseBias([(2, 3)], bias=1.5, top_k=2)
    input_ids = torch.tensor([[9]])
    scores = torch.tensor([[0.0, 0.1, 0.9, 0.2, 1.0]])
    out = proc(input_ids, scores.clone())
    assert abs(out[0, 2].item() - (scores[0, 2].item() + 1.5)) < 1e-6

    proc_top1 = AcousticTopKPhraseBias([(2, 3)], bias=1.5, top_k=1)
    out_top1 = proc_top1(input_ids, scores.clone())
    assert out_top1[0, 2].item() == scores[0, 2].item()


def test_phrase_continuation_is_boosted():
    proc = AcousticTopKPhraseBias([(2, 3, 4)], bias=0.75, top_k=3)
    scores = torch.tensor([[0.0, 0.1, 0.7, 0.8, 0.9]])
    out = proc(torch.tensor([[8, 2, 3]]), scores.clone())
    assert abs(out[0, 4].item() - (scores[0, 4].item() + 0.75)) < 1e-6


def test_unrelated_low_rank_hotword_is_not_invented():
    proc = AcousticTopKPhraseBias([(1, 2)], bias=10.0, top_k=2)
    scores = torch.tensor([[0.0, -5.0, -4.0, 2.0, 1.0]])
    out = proc(torch.tensor([[7]]), scores.clone())
    assert out[0, 1].item() == scores[0, 1].item()


def test_load_line_file_and_json():
    with mock.patch(
        "builtins.open", return_value=io.StringIO("# comment\n空调\n打开\n空调\n")
    ):
        assert load_phrases("phrases.txt") == ["空调", "打开"]

    with mock.patch(
        "builtins.open", return_value=io.StringIO('["电视", "关闭"]')
    ):
        assert load_phrases("phrases.json") == ["电视", "关闭"]


def test_stratified_selection_covers_commands():
    rows = [
        {"target_ref": ref, "overlap_ratio": overlap, "file": f"{ref}_{overlap}.wav"}
        for ref in ("a", "b", "c")
        for overlap in (0.0, 0.5, 1.0)
    ]
    selected = select_stratified(rows, 6)
    assert [row["target_ref"] for row in selected[:3]] == ["a", "b", "c"]
    assert len({row["target_ref"] for row in selected}) == 3
    assert len({row["overlap_ratio"] for row in selected}) == 3


if __name__ == "__main__":
    test_phrase_start_is_boosted_only_inside_top_k()
    test_phrase_continuation_is_boosted()
    test_unrelated_low_rank_hotword_is_not_invented()
    test_load_line_file_and_json()
    test_stratified_selection_covers_commands()
    print("ALL PASS")
