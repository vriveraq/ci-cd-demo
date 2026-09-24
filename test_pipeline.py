import os
import joblib
import pandas as pd
import pytest
from pydantic import ValidationError

from data_pipeline import (
    TransactionRecord,
    load_and_validate_fraud_data,
    train_fraud_detection_model,
)


def test_fraud_dataset_ingestion():
    """Validates local dataset generation and row counts."""
    df = load_and_validate_fraud_data()
    assert not df.empty
    assert len(df) == 1000
    assert "transaction_amount" in df.columns
    assert set(df["is_fraud"].unique()).issubset({0, 1})


def test_pydantic_schema_rejects_corrupted_data():
    """Ensures quality gate catches corrupted raw records."""
    with pytest.raises(ValidationError):
        TransactionRecord(
            account_age=12,
            transaction_amount=-150.00,  # Negative amount triggers Pydantic error
            payment_method="creditcard",
            is_fraud=1,
        )


def test_pipeline_training_and_inference():
    """Tests Scikit-Learn Pipeline fitting, artifact creation, and raw inference."""
    df = load_and_validate_fraud_data()
    pipeline = train_fraud_detection_model(df)

    # 1. Verify artifact exists
    assert os.path.exists("fraud_pipeline.joblib")

    # 2. Check baseline accuracy score on pipeline
    X = df[["payment_method", "account_age", "transaction_amount"]]
    y = df["is_fraud"]
    accuracy = pipeline.score(X, y)
    assert accuracy >= 0.80

    # 3. Test raw prediction using loaded pipeline artifact
    loaded_pipeline = joblib.load("fraud_pipeline.joblib")
    raw_sample = pd.DataFrame(
        [
            {
                "payment_method": "paypal",
                "account_age": 2,
                "transaction_amount": 450.00,
            }
        ]
    )
    prediction = loaded_pipeline.predict(raw_sample)
    assert prediction[0] in [0, 1]