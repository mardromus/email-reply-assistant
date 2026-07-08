import numpy as np
from retrieval.embedding import EmbeddingModel

def test_embedding_model_initialization():
    model = EmbeddingModel()
    assert model.model_name == "all-MiniLM-L6-v2"
    assert model.dimension == 384

def test_embedding_encoding():
    model = EmbeddingModel()
    texts = ["Hello world", "This is a test"]
    embeddings = model.encode(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32

def test_similarity():
    model = EmbeddingModel()
    emb1 = model.encode_single("How do I cancel my account?")
    emb2 = model.encode_single("Can I get a refund and close my account?")
    emb3 = model.encode_single("What is the weather like today?")
    
    sim_similar = model.similarity(emb1, emb2)
    sim_different = model.similarity(emb1, emb3)
    
    assert sim_similar > sim_different
