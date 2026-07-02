"""Unit tests for the RAG rank-fusion logic (no heavy dependencies)."""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.fusion import reciprocal_rank_fusion


def test_rrf_rewards_agreement_across_lists():
    # "b" is ranked highly by both retrievers; it should win overall.
    dense = ["a", "b", "c"]
    sparse = ["b", "d", "a"]
    fused = reciprocal_rank_fusion([dense, sparse])
    keys = [k for k, _ in fused]
    assert keys[0] == "b"
    # every unique key appears exactly once
    assert set(keys) == {"a", "b", "c", "d"}


def test_rrf_single_list_preserves_order():
    fused = reciprocal_rank_fusion([["x", "y", "z"]])
    assert [k for k, _ in fused] == ["x", "y", "z"]


def test_rrf_scores_descending():
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)
    assert fused[0][0] == "a"  # appears in both lists


def test_rrf_weights_bias_towards_a_list():
    dense = ["a", "b"]
    sparse = ["b", "a"]
    # Heavily weight the dense list; its top ("a") should lead.
    fused = reciprocal_rank_fusion([dense, sparse], weights=[5.0, 1.0])
    assert fused[0][0] == "a"


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []
