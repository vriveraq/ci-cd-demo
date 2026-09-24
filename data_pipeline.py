import os
import joblib
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class TransactionRecord(BaseModel):
    account_age: int = Field(ge=0, description="Account age in months cannot be negative")
    transaction_amount: float = Field(gt=0, description="Transaction amount must be positive")
    payment_method: str
    is_fraud: int = Field(ge=0, le=1, description="Binary classification label")

def fetch_fraud_data() -> pd.DataFrame:
    # Fetches real Credit Card Fraud dataset directly from OpenML (Dataset ID: 42175 / creditcard)
    print("Fetching dataset from OpenML...")
    data = fetch_openml(data_id=42175, as_frame=True, parser="auto")
    df = data.frame.sample(n=1000, random_state=42) # Sample 1000 rows for fast CI runs
    return df

def load_and_validate_fraud_data(csv_path="payment_fraud.csv") -> pd.DataFrame:
    """Ensures dataset exists, parses schema, and validates data quality with Pydantic."""
    if not os.path.exists(csv_path):
        df = fetch_fraud_data()
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    validated_records = []
    for _, row in df.iterrows():
        record = TransactionRecord(
            account_age=int(row["accountAge"]),
            transaction_amount=float(row["transactionAmount"]),
            payment_method=str(row["paymentMethod"]),
            is_fraud=int(row["label"]),
        )
        validated_records.append(record.model_dump())

    return pd.DataFrame(validated_records)


def train_fraud_detection_model(df: pd.DataFrame) -> Pipeline:
    """Trains an end-to-end ML Pipeline using OneHotEncoder and RandomForestClassifier."""
    
    # Define categorical and numerical features
    categorical_features = ["payment_method"]
    numeric_features = ["account_age", "transaction_amount"]

    # Preprocessor using OneHotEncoder instead of pd.get_dummies()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("num", "passthrough", numeric_features),
        ]
    )

    # Scikit-Learn Pipeline combining preprocessing and model
    clf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=25, random_state=42)),
        ]
    )

    X = df[["payment_method", "account_age", "transaction_amount"]]
    y = df["is_fraud"]

    # Train full pipeline
    clf_pipeline.fit(X, y)

    # Save single pipeline artifact (contains transformer + model)
    joblib.dump(clf_pipeline, "fraud_pipeline.joblib")
    return clf_pipeline