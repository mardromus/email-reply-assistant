import pytest
import os
import shutil
from retrieval.vector_store import EmailVectorStore
from retrieval.search import retrieve_similar, set_store

@pytest.fixture
def temp_chroma_dir(tmp_path):
    # Set the chroma directory to a temporary path for testing
    import config
    original = config.get_settings().chroma_persist_dir
    temp_dir = str(tmp_path / "chroma")
    config.get_settings().chroma_persist_dir = temp_dir
    yield temp_dir
    # Cleanup
    config.get_settings().chroma_persist_dir = original
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def test_vector_store_build_index(temp_chroma_dir, sample_emails_df):
    store = EmailVectorStore()
    store.build_index(sample_emails_df)
    
    assert store.get_collection_count() == 3
    
    # Query test
    results = store.query(["refund my subscription"], n_results=1)
    assert len(results["ids"][0]) == 1
    assert "Refund" in results["metadatas"][0][0]["category"]

def test_search_retrieve_similar(temp_chroma_dir, sample_emails_df):
    store = EmailVectorStore()
    store.build_index(sample_emails_df)
    set_store(store)
    
    results = retrieve_similar("I forgot my password", top_k=1)
    assert len(results) == 1
    assert results[0].category == "Technical Issue"
    assert "reset your password" in results[0].reply_text
