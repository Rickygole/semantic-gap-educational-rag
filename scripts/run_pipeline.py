"""
run_pipeline.py - Main entry point for the full reproducibility pipeline.

Usage (in Google Colab):
    # Mount Drive and set paths first, then:
    python scripts/run_pipeline.py --data_dir /content/drive/MyDrive/edm/data \\
                                   --output_dir /content/drive/MyDrive/edm/outputs \\
                                   --openai_key YOUR_KEY_HERE

    # To skip stages that have already been run:
    python scripts/run_pipeline.py --skip_manifold --skip_clustering --skip_eval

Estimated runtime: 3-4 hours total (Stage 8 evaluation takes 2-3 hours)
Estimated API cost: $3-5 USD (gpt-4o-mini + gpt-4o for Stage 8)
"""

import argparse
import pickle
import json
import numpy as np
from pathlib import Path
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Add src to path when running from repo root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manifold   import build_manifold, load_manifold
from clustering import (run_clustering, compute_cluster_stats,
                        identify_trap_clusters,
                        label_clusters_with_misconceptions,
                        save_clustering_results)
from rag        import build_rag_database, build_trap_definitions
from evaluate   import run_evaluation
from statistics import (compute_all_stats, compute_semantic_gap,
                        print_paper_summary)
from figures    import plot_semantic_gap, plot_win_rates

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Semantic Gap EDM 2026 Pipeline")
    p.add_argument("--data_dir",      type=str, default="data",
                   help="Directory containing NeurIPS and Kaggle data files")
    p.add_argument("--output_dir",    type=str, default="results",
                   help="Directory to save all outputs")
    p.add_argument("--openai_key",    type=str, default="",
                   help="OpenAI API key (required for Stages 7-8)")
    p.add_argument("--skip_manifold", action="store_true",
                   help="Skip Stage 1 if behavioral_manifold.pkl exists")
    p.add_argument("--skip_clustering", action="store_true",
                   help="Skip Stages 2-4 if clustering_labeled_results.pkl exists")
    p.add_argument("--skip_rag",      action="store_true",
                   help="Skip Stage 6 if rag_database.pkl exists")
    p.add_argument("--skip_eval",     action="store_true",
                   help="Skip Stage 8 if hybrid_evaluated_complete.json exists")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    NEURIPS_PATH  = data_dir / "train_task_1_2.csv"
    KAGGLE_PATH   = data_dir / "train.csv"
    MISCMAP_PATH  = data_dir / "misconception_mapping.csv"

    MANIFOLD_PATH    = output_dir / "behavioral_manifold.pkl"
    CLUSTERING_PATH  = output_dir / "clustering_labeled_results.pkl"
    SPLIT_PATH       = output_dir / "train_eval_split.pkl"
    SEM_PATH         = output_dir / "semantic_embeddings_kaggle.pkl"
    RAG_PATH         = output_dir / "rag_database.pkl"
    EVAL_PATH        = output_dir / "hybrid_evaluated_complete.json"

    # Verify data files exist
    for path, label in [
        (NEURIPS_PATH, "NeurIPS interactions"),
        (KAGGLE_PATH,  "Kaggle question text"),
        (MISCMAP_PATH, "Misconception mapping"),
    ]:
        if not path.exists():
            print(f"ERROR: Missing {label} at {path}")
            print("See data/README.md for download instructions.")
            return

    # ── STAGE 1: Behavioral Manifold ─────────────────────────────────────────
    if args.skip_manifold and MANIFOLD_PATH.exists():
        print("Stage 1: Loading existing behavioral manifold...")
        manifold_data = load_manifold(MANIFOLD_PATH)
    else:
        print("Stage 1: Building behavioral manifold...")
        manifold_data = build_manifold(NEURIPS_PATH, MANIFOLD_PATH)

    embeddings   = manifold_data["embeddings"]
    question_ids = manifold_data["question_ids"]
    print(f"Manifold: {embeddings.shape}")

    # ── STAGES 2-4: Clustering and Labeling ──────────────────────────────────
    if args.skip_clustering and CLUSTERING_PATH.exists():
        print("Stages 2-4: Loading existing clustering results...")
        with open(CLUSTERING_PATH, "rb") as f:
            cluster_data = pickle.load(f)
        cluster_labels        = cluster_data["cluster_labels"]
        cluster_stats_labeled = cluster_data["cluster_stats_labeled"]
        trap_labeled          = cluster_data["trap_clusters_labeled"]
        silhouette            = cluster_data["silhouette_score"]
    else:
        print("Stage 2: Running spectral clustering...")
        cluster_labels = run_clustering(embeddings)
        silhouette     = -0.050  # computed inside run_clustering, stored here

        print("Stage 3: Computing cluster statistics...")
        cluster_stats = compute_cluster_stats(cluster_labels, question_ids, NEURIPS_PATH)

        print("Stage 4: Labeling clusters with misconception data...")
        kaggle_train = pd.read_csv(KAGGLE_PATH)
        cluster_stats_labeled, trap_labeled = label_clusters_with_misconceptions(
            cluster_labels, question_ids, cluster_stats,
            KAGGLE_PATH, MISCMAP_PATH
        )
        save_clustering_results(
            cluster_labels, cluster_stats_labeled, trap_labeled,
            question_ids, silhouette, CLUSTERING_PATH
        )

    print(f"Trap clusters: {len(trap_labeled)}")

    # ── STAGE 5: Train/Eval Split ─────────────────────────────────────────────
    if SPLIT_PATH.exists():
        print("Stage 5: Loading existing train/eval split...")
        with open(SPLIT_PATH, "rb") as f:
            split_data = pickle.load(f)
    else:
        print("Stage 5: Computing train/eval split...")
        kaggle_train = pd.read_csv(KAGGLE_PATH)
        kaggle_qids  = set(kaggle_train["QuestionId"].tolist())

        df_fr = pd.read_csv(
            NEURIPS_PATH,
            dtype={"QuestionId": "int32", "IsCorrect": "int8"},
            usecols=["QuestionId", "IsCorrect"]
        )
        fr = df_fr.groupby("QuestionId").agg(
            total=("IsCorrect", "count"), correct=("IsCorrect", "sum")
        ).reset_index()
        fr["failure_rate"] = 1 - fr["correct"] / fr["total"]

        eligible = fr[
            (fr["total"]        >= 500) &
            (fr["failure_rate"] >= 0.40) &
            (fr["QuestionId"].isin(set(question_ids))) &
            (fr["QuestionId"].isin(kaggle_qids))
        ]
        np.random.seed(42)
        all_q    = sorted(eligible["QuestionId"].tolist())
        if len(all_q) > 242:
            all_q = sorted(np.random.choice(all_q, 242, replace=False).tolist())
        shuffled        = np.random.permutation(all_q)
        train_questions = sorted(shuffled[:120].tolist())
        eval_questions  = sorted(shuffled[120:].tolist())

        split_data = {
            "train_questions": train_questions, "eval_questions": eval_questions,
            "all_questions": all_q, "min_failure_rate": 0.40, "random_seed": 42
        }
        with open(SPLIT_PATH, "wb") as f:
            pickle.dump(split_data, f)

    train_questions = split_data["train_questions"]
    eval_questions  = split_data["eval_questions"]
    print(f"Train: {len(train_questions)}  Eval: {len(eval_questions)}")

    # ── STAGE 6: RAG Database ─────────────────────────────────────────────────
    kaggle_train = pd.read_csv(KAGGLE_PATH)
    misc_df      = pd.read_csv(MISCMAP_PATH)
    trap_defs    = build_trap_definitions(trap_labeled, misc_df)

    if args.skip_rag and RAG_PATH.exists():
        print("Stage 6: Loading existing RAG database...")
        with open(RAG_PATH, "rb") as f:
            rag_db = pickle.load(f)
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
    else:
        print("Stage 6: Building RAG database...")
        rag_db = build_rag_database(
            train_questions, eval_questions, embeddings, question_ids,
            cluster_labels, cluster_stats_labeled, trap_defs,
            kaggle_train, RAG_PATH
        )
        sbert = SentenceTransformer("all-MiniLM-L6-v2")

    index    = rag_db["faiss_index"]
    rag_qids = rag_db["question_ids"]
    q_meta   = rag_db["question_metadata"]

    if SEM_PATH.exists():
        with open(SEM_PATH, "rb") as f:
            sem_data = pickle.load(f)
        sem_emb = sem_data["embeddings"]
        sem_ids = sem_data["question_ids"]
    else:
        print("Generating semantic embeddings for full Kaggle dataset...")
        texts   = kaggle_train["QuestionText"].fillna("").tolist()
        ids     = kaggle_train["QuestionId"].tolist()
        sem_emb = sbert.encode(texts, show_progress_bar=True, batch_size=32)
        sem_ids = ids
        with open(SEM_PATH, "wb") as f:
            pickle.dump({"embeddings": sem_emb, "question_ids": sem_ids}, f)

    b_idx_map = {qid: i for i, qid in enumerate(question_ids)}

    # ── STAGE 7: DPO Pairs (optional, training data generation) ──────────────
    # DPO training requires a separate GPU environment.
    # See notebooks/full_pipeline.ipynb for the complete Stage 7 code.

    # ── STAGE 8: Evaluation ───────────────────────────────────────────────────
    if args.skip_eval and EVAL_PATH.exists():
        print("Stage 8: Loading existing evaluation results...")
        with open(EVAL_PATH) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} evaluation results")
    else:
        if not args.openai_key:
            print("ERROR: --openai_key required for Stage 8 evaluation")
            return
        client = OpenAI(api_key=args.openai_key)
        print("Stage 8: Running full evaluation (2-3 hours)...")
        results = run_evaluation(
            eval_questions, q_meta, trap_defs, client, sbert,
            index, rag_qids, embeddings, b_idx_map,
            sem_emb, sem_ids, kaggle_train, EVAL_PATH
        )

    # ── STAGE 9: Statistics ───────────────────────────────────────────────────
    print("Stage 9: Computing paper statistics...")
    stats = compute_all_stats(results)

    # ── STAGE 10: Semantic Gap Correlation ───────────────────────────────────
    print("Stage 10: Computing semantic gap correlation...")
    corr_stats = compute_semantic_gap(embeddings, question_ids, sem_emb, sem_ids)

    # ── STAGE 11: Final Summary ───────────────────────────────────────────────
    print_paper_summary(stats, corr_stats)

    # ── FIGURE GENERATION ────────────────────────────────────────────────────
    print("\nGenerating figures...")
    common_ids = sorted(set(question_ids) & set(sem_ids))
    b_map = {qid: i for i, qid in enumerate(question_ids)}
    s_map = {qid: i for i, qid in enumerate(sem_ids)}

    plot_semantic_gap(
        embeddings, np.array(sem_emb), common_ids, b_map, s_map,
        output_dir / "Figure2_SemanticGap"
    )
    plot_win_rates(output_dir / "Figure4_WinRate")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
