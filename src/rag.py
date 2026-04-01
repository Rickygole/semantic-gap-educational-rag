"""
rag.py - Build the RAG database and implement the four tutoring architectures.

The MAG architecture implements Semantic Primacy:
  Stage 1 (Semantic Filter): Retrieve top-20 semantically similar questions
           using S-BERT cosine similarity. This ensures topical relevance
           and prevents Topical Drift.
  Stage 2 (Behavioral Personalization): Within the semantic candidate set,
           select top-5 by behavioral similarity in the 50D SVD manifold.
           This surfaces latent misconceptions invisible to semantic search.

Why this order matters: behavioral-only retrieval systematically selects
topically irrelevant questions with similar difficulty profiles, transforming
neutral noise into structured misdirection (see paper Section 4.2).
"""

import pickle
import json
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def build_rag_database(
    train_questions: list,
    eval_questions: list,
    embeddings: np.ndarray,
    question_ids: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_stats_labeled: pd.DataFrame,
    trap_definitions: dict,
    kaggle_train_df: pd.DataFrame,
    output_path: Path,
    model_name: str = "all-MiniLM-L6-v2"
) -> dict:
    """
    Encode all 242 questions with S-BERT and build a FAISS index.

    The FAISS index enables efficient nearest-neighbor retrieval for
    System B (semantic RAG) and Stage 1 of System D (MAG).

    Args:
        train_questions:      list of 120 training question IDs
        eval_questions:       list of 122 evaluation question IDs
        embeddings:           (27613, 50) behavioral manifold
        question_ids:         question IDs corresponding to embeddings rows
        cluster_labels:       cluster assignment per question
        cluster_stats_labeled: cluster statistics with failure rates
        trap_definitions:     {cluster_id: misconception_name}
        kaggle_train_df:      Kaggle question text DataFrame
        output_path:          where to save rag_database.pkl
        model_name:           S-BERT model (all-MiniLM-L6-v2, 384 dimensions)

    Returns:
        rag_database dict with FAISS index and question metadata
    """
    print(f"Loading S-BERT model ({model_name})...")
    sbert     = SentenceTransformer(model_name)
    b_idx_map = {qid: i for i, qid in enumerate(question_ids)}

    all_questions     = train_questions + eval_questions
    question_metadata = {}

    for q_id in all_questions:
        q_row = kaggle_train_df[kaggle_train_df["QuestionId"] == q_id]
        if len(q_row) == 0:
            continue
        q_row    = q_row.iloc[0]
        beh_idx  = b_idx_map.get(q_id)
        if beh_idx is None:
            continue
        trap_cluster = int(cluster_labels[beh_idx])
        trap_stats   = cluster_stats_labeled[
            cluster_stats_labeled["cluster_id"] == trap_cluster
        ].iloc[0]
        mc_ids = [
            int(q_row[c]) for c in [
                "MisconceptionAId", "MisconceptionBId",
                "MisconceptionCId", "MisconceptionDId"
            ]
            if c in q_row and pd.notna(q_row[c])
        ]
        question_metadata[q_id] = {
            "question_text":     q_row["QuestionText"],
            "correct_answer":    q_row["CorrectAnswer"],
            "trap_cluster":      trap_cluster,
            "failure_rate":      float(trap_stats["failure_rate"]),
            "density":           int(trap_stats["density"]),
            "misconception_ids": mc_ids,
            "is_training":       q_id in train_questions
        }

    rag_texts = [question_metadata[q]["question_text"]
                 for q in all_questions if q in question_metadata]
    rag_qids  = [q for q in all_questions if q in question_metadata]

    print(f"Encoding {len(rag_texts)} questions with S-BERT...")
    rag_emb = sbert.encode(rag_texts, show_progress_bar=True, batch_size=32)

    # FAISS L2 index for efficient nearest-neighbor search
    index = faiss.IndexFlatL2(rag_emb.shape[1])
    index.add(rag_emb.astype("float32"))

    rag_database = {
        "embeddings":        rag_emb,
        "faiss_index":       index,
        "question_ids":      rag_qids,
        "question_metadata": question_metadata,
        "train_questions":   train_questions,
        "eval_questions":    eval_questions,
        "model_name":        model_name
    }

    with open(output_path, "wb") as f:
        pickle.dump(rag_database, f)
    print(f"Saved: {output_path} ({index.ntotal} vectors)")
    return rag_database


def build_trap_definitions(
    trap_labeled: pd.DataFrame,
    misc_df: pd.DataFrame
) -> dict:
    """
    Build a mapping from cluster_id to misconception name for all clusters.

    For clusters without a dominant misconception in the expert labels,
    a generic fallback label is used.
    """
    trap_definitions = {}
    for _, row in trap_labeled.iterrows():
        cid = row["cluster_id"]
        if pd.notna(row.get("dominant_misconception")):
            mc_id  = int(row["dominant_misconception"])
            mc_row = misc_df[misc_df["MisconceptionId"] == mc_id]
            trap_definitions[cid] = (
                mc_row.iloc[0]["MisconceptionName"]
                if len(mc_row) > 0 else f"Trap {cid}"
            )
        else:
            trap_definitions[cid] = f"Trap {cid}"
    return trap_definitions
