"""
manifold.py - Build the 50-dimensional SVD behavioral manifold.

Why SVD: We intentionally use linear SVD rather than deep learning methods
to audit the specific assumptions of real-time production systems, which
default to matrix factorization for computational efficiency at scale.

The resulting manifold embeds each question as a 50-dimensional vector
capturing latent patterns of student success and failure across the full
interaction dataset.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def build_manifold(interactions_path: Path, output_path: Path) -> dict:
    """
    Build a 50-dimensional SVD behavioral manifold from student interactions.

    The interaction matrix M is binary: M[student, question] = 1 if correct.
    SVD decomposes M into low-dimensional representations capturing latent
    patterns of how students fail and succeed across questions.

    Args:
        interactions_path: Path to NeurIPS 2020 train_task_1_2.csv
        output_path: Path to save behavioral_manifold.pkl

    Returns:
        dict with keys: embeddings, question_ids, explained_variance_ratio
    """
    print(f"Loading interactions from {interactions_path}...")
    df = pd.read_csv(
        interactions_path,
        dtype={"QuestionId": "int32", "UserId": "int32", "IsCorrect": "int8"},
        usecols=["UserId", "QuestionId", "IsCorrect"]
    )
    print(f"Loaded {len(df):,} interactions")

    # Build student-question index maps
    users     = df["UserId"].unique()
    questions = df["QuestionId"].unique()
    u2i = {u: i for i, u in enumerate(users)}
    q2i = {q: i for i, q in enumerate(questions)}
    df["ui"] = df["UserId"].map(u2i)
    df["qi"] = df["QuestionId"].map(q2i)

    # Construct sparse binary interaction matrix (users x questions)
    M = csr_matrix(
        (df["IsCorrect"].values, (df["ui"].values, df["qi"].values)),
        shape=(len(users), len(questions)),
        dtype=np.float32
    )
    print(f"Interaction matrix: {M.shape[0]:,} students x {M.shape[1]:,} questions")

    # Apply SVD to the transpose (questions x users) so each question
    # gets a 50-dimensional embedding in behavioral space
    print("Running SVD (50 components)...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    Q   = svd.fit_transform(M.T)

    # L2-normalize so cosine similarity equals dot product
    Q = normalize(Q, norm="l2")
    print(f"Explained variance: {svd.explained_variance_ratio_.sum():.2%}")

    manifold_data = {
        "embeddings":               Q,
        "question_ids":             questions,
        "question_to_idx":          q2i,
        "n_components":             50,
        "explained_variance_ratio": svd.explained_variance_ratio_,
        "singular_values":          svd.singular_values_
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(manifold_data, f)
    print(f"Saved: {output_path}")
    return manifold_data


def load_manifold(path: Path) -> dict:
    """Load a previously saved behavioral manifold."""
    with open(path, "rb") as f:
        return pickle.load(f)
