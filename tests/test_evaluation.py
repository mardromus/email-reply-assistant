import pytest
from evaluation.semantic_similarity import SemanticSimilarityMetric
from evaluation.readability import ReadabilityMetric

def test_semantic_similarity_metric():
    metric = SemanticSimilarityMetric()
    
    # High similarity
    res1 = metric.evaluate(
        generated="I have refunded your order.",
        reference="Your order has been refunded.",
        email="Refund request"
    )
    assert res1.score > 0.8
    
    # Low similarity
    res2 = metric.evaluate(
        generated="Please reset your password.",
        reference="Your order has been refunded.",
        email="Refund request"
    )
    assert res2.score < 0.5

def test_readability_metric():
    metric = ReadabilityMetric()
    
    res = metric.evaluate(
        generated="This is a simple sentence. It is easy to read.",
        reference="",
        email=""
    )
    
    assert res.score > 0.5
    assert not res.penalized
    assert "Flesch" in res.details
