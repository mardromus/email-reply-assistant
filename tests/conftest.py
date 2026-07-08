import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_emails_df():
    data = {
        "id": ["1", "2", "3"],
        "email": [
            "Where is my order? Tracking says it was delivered but I don't see it.",
            "Can I get a refund for my subscription?",
            "How do I reset my password?"
        ],
        "reply": [
            "I'm sorry to hear your order is missing. I've opened an investigation with the carrier.",
            "I've processed a refund for your subscription. It should appear in 3-5 days.",
            "Click the 'Forgot Password' link on the login page to reset your password."
        ],
        "category": ["Support", "Refund", "Technical Issue"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_embedding_result():
    return np.random.rand(3, 384).astype(np.float32)
