import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between 2 vectors, by hand."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm)


def main() -> None:
    # Load the model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Five normal LLM style factual answers.
    baseline = [
        "Paris is the capital of France.",
        "Water boils at 100 degree celsisus",
        "The Earth orbitsthe Sun.",
        "Potosynthesis converts sunlightinto chemical energy",
        "Shakespeare wrote Hamlet",
    ]

    # Embed all 5 -> shape (5,384)
    baseline_vectors = model.encode(baseline)

    # Centroid = mean across rows -> shape(384,). One "average" vector
    centroid = np.mean(baseline_vectors, axis=0)

    # The new input we want to classify (change this and rerun the test)
    new_text = "Mount Everest is the tallest mountain on Earth."

    # Embed the new input -> shape(384,).
    new_vector = model.encode(new_text)

    # Anomaly score =1 - cosine. Higher = more anomalies
    score = 1.0 - cosine_similarity(new_vector, centroid)

    # Threshold- based decision
    threshold = 0.35
    label = "ANOMALY" if score > threshold else "NORMAL"

    print(f" Input : {new_text}")
    print(f" Score : {score:.3f}")
    print(f" Threshold : {threshold}")
    print(f"Verdict : {label}")


if __name__ == "__main__":
    main()
