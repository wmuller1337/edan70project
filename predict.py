"""Predict the climate-denial label for a text file using the trained model.

Usage:
    python predict.py path/to/article.txt
    python predict.py path/to/article.txt --model ./model/denial_clf.joblib

The script loads the saved classifier bundle (produced by running files.py),
embeds the input text with the SAME multilingual model used in training
(truncating to the same number of characters), and prints the predicted label
plus the denial probability. It works on text in any language the embedding
model supports.
"""

import argparse

import numpy as np

from evaluation import load_model

# Lazily-initialised embedder (cached across calls)
_EMBEDDER = None


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_embedder(embedding_model):
    """Load (once) the SentenceTransformer used for embedding."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(embedding_model, device=_device())
    return _EMBEDDER


def embed_texts(texts, embedding_model, max_chars=2000):
    """Embed texts the same way training did (truncate, then encode).

    The harrier model normalises internally (its own Normalize module), matching
    how ChromaDB embedded the training documents, so no extra normalisation is
    applied here.
    """
    embedder = _get_embedder(embedding_model)
    truncated = [str(t)[:max_chars] for t in texts]
    vectors = embedder.encode(truncated, convert_to_numpy=True)
    return np.asarray(vectors)


def predict_texts(texts, model_path="./model/denial_clf.joblib"):
    """Return predictions for a list of texts.

    Each result: {label, label_name, denial_prob}.
    """
    bundle = load_model(model_path)
    model = bundle["model"]
    target_names = bundle["target_names"]
    embedding_model = bundle["embedding_model"]
    max_chars = bundle.get("max_chars", 2000)

    X = embed_texts(texts, embedding_model, max_chars=max_chars)
    labels = model.predict(X)
    probs = model.predict_proba(X)

    results = []
    for label, prob in zip(labels, probs):
        label = int(label)
        results.append({
            "label": label,
            "label_name": target_names[label],
            "denial_prob": float(prob[1]),
        })
    return results


def predict_file(txt_path, model_path="./model/denial_clf.joblib"):
    """Read a .txt file, classify it, print and return the result."""
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        text = f.read().strip()

    if not text:
        raise ValueError(f"{txt_path} is empty")

    result = predict_texts([text], model_path=model_path)[0]
    print(f"File:        {txt_path}")
    print(f"Prediction:  {result['label_name']}")
    print(f"Denial prob: {result['denial_prob']:.3f}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Classify a .txt article as climate denial / not-denial."
    )
    parser.add_argument("txt_file", help="Path to a UTF-8 text file (an article)")
    parser.add_argument(
        "--model",
        default="./model/denial_clf.joblib",
        help="Path to the saved model bundle (default: ./model/denial_clf.joblib)",
    )
    args = parser.parse_args()
    predict_file(args.txt_file, model_path=args.model)


if __name__ == "__main__":
    main()
