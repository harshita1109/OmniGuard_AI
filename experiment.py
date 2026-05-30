import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, computed by hand."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm)


def main() -> None:
    # 1. Load the model (downloads ~80 MB on first run, then cached).
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 2. Three sentences: two similar in meaning, one unrelated.
    sentences = [
        "The cat sat on the mat",
        "A feline rested on the rug",
        "Quantum chromodynamics is hard",
    ]

    # 3. Embed all three. Output shape: (3, 384).
    embeddings = model.encode(sentences)
    print(f"Embedding shape: {embeddings.shape}")  # expect (3, 384)

    # 4. Compute cosine similarity for each pair, by hand.
    pairs = [
        ("cat-mat vs feline-rug", embeddings[0], embeddings[1]),
        ("cat-mat vs quantum", embeddings[0], embeddings[2]),
        ("feline-rug vs quantum", embeddings[1], embeddings[2]),
    ]

    for label, a, b in pairs:
        score = cosine_similarity(a, b)
        print(f"{label:30s} -> {score:.3f}")


if __name__ == "__main__":
    main()
