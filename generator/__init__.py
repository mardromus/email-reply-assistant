"""
Response Generation module for the AI Email Response System.

Provides LLM interaction, prompt engineering, response generation,
and the full RAG (Retrieval-Augmented Generation) pipeline.
"""

from generator.llm import CerebrasLLM
from generator.generate import GeneratedResponse, generate_response, generate_batch
from generator.rag_pipeline import RAGPipeline, PipelineResult

__all__ = [
    "CerebrasLLM",
    "GeneratedResponse",
    "generate_response",
    "generate_batch",
    "RAGPipeline",
    "PipelineResult",
]
