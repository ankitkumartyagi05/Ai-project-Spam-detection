import argparse, os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd, joblib

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    ns = parser.parse_args(args)

    # Simple train for spam detection
    df = pd.read_csv("data/raw/train.csv")
    X, y = df.text, df.label
    vec = TfidfVectorizer()
    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline([("vec", vec), ("clf", clf)])
    pipe.fit(X, y)

    preds = pipe.predict(X)
    print(classification_report(y, preds))

    os.makedirs("models/artifacts", exist_ok=True)
    joblib.dump(pipe, "models/artifacts/spamshield_tfidf.joblib")
    print("✅ Model saved at models/artifacts/spamshield_tfidf.joblib")

if __name__ == "__main__":
    main()
