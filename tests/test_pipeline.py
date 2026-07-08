import pytest
from unittest.mock import patch, MagicMock
from generator.rag_pipeline import RAGPipeline
from generator.generate import GeneratedResponse

@patch('generator.rag_pipeline.CerebrasLLM')
@patch('generator.rag_pipeline.retrieve_similar')
def test_rag_pipeline_process(mock_retrieve, mock_llm_class, sample_emails_df):
    # Mock retrieval
    from retrieval.search import RetrievedEmail
    mock_retrieved = RetrievedEmail(
        email_id="2",
        email_text="Can I get a refund?",
        reply_text="Yes, refund processed.",
        similarity_score=0.9,
        metadata={"category": "Refund"}
    )
    mock_retrieve.return_value = [mock_retrieved]
    
    # Mock LLM
    mock_llm_instance = MagicMock()
    mock_llm_instance.generate.return_value = "I've processed your refund."
    mock_llm_class.return_value = mock_llm_instance
    
    # Run pipeline
    pipeline = RAGPipeline()
    result = pipeline.process(email="I want a refund for my order", top_k=1)
    
    assert result.email == "I want a refund for my order"
    assert result.generated_response == "I've processed your refund."
    assert len(result.retrieved_examples) == 1
    assert result.retrieved_examples[0]["similarity_score"] == 0.9
    
    mock_retrieve.assert_called_once()
    mock_llm_instance.generate.assert_called_once()
