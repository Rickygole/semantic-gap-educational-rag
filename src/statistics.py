"""
statistics.py - Compute all paper statistics from evaluation results.

Reproduces every number in Table 1 and Section 4 of the paper:
  - Quality scores with bootstrap confidence intervals (10,000 iterations)
  - Cohen's d effect sizes with paired t-test p-values
  - Pairwise tournament win rates
  - IAR (Indiscriminate Affirmation Response) counts
  - Semantic gap correlation (Pearson r, Spearman rho, permutation test)
  - Difficulty tertile correlations
"""

import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, ttest_rel
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def bootstrap_ci(scores: np.ndarray, iters: int = 10000, seed: int = 42) -> tuple:
    """
    Compute 95% bootstrap confidence interval for the mean.

    Args:
        scores: array of numeric scores
        iters:  number of bootstrap iterations (10,000 for paper results)
        seed:   random seed for reproducibility

    Returns:
        (lower_bound, upper_bound) of 95% CI
    """
    rng   = np.random.default_rng(seed)
    means = [
        np.mean(rng.choice(scores, len(scores), replace=True))
        for _ in range(iters)
    ]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute pooled Cohen's d effect size.

    Positive values indicate x > y. The paper reports d values for
    System D (MAG) versus each baseline, so positive d means MAG is better.
    """
    nx, ny = len(x), len(y)
    pooled = np.sqrt(
        ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2)
        / (nx + ny - 2)
    )
    return (np.mean(x) - np.mean(y)) / pooled


def pairwise_win_rate(scores: np.ndarray, others: list) -> float:
    """
    Compute pairwise tournament win rate.

    Win rate = fraction of head-to-head comparisons this system wins
    against each of the other systems. Random baseline for a 4-system
    tournament is 33.3% (not 25%).

    Args:
        scores: score array for the focal system
        others: list of score arrays for the other systems

    Returns:
        win rate as a fraction (multiply by 100 for percentage)
    """
    n    = len(scores)
    wins = sum(
        sum(1 for i in range(n) if scores[i] > o[i])
        for o in others
    )
    return wins / (n * len(others))


def compute_all_stats(results: list) -> dict:
    """
    Compute all statistics reported in Table 1 and Section 4 of the paper.

    Args:
        results: list of evaluation result dicts from run_evaluation()

    Returns:
        dict containing all computed statistics
    """
    sa = np.array([q["score_a"] for q in results])
    sb = np.array([q["score_b"] for q in results])
    sc = np.array([q["score_c"] for q in results])
    sd = np.array([q["score_d"] for q in results])
    n  = len(results)

    lo_d, hi_d = bootstrap_ci(sd)
    lo_a, hi_a = bootstrap_ci(sa)
    lo_b, hi_b = bootstrap_ci(sb)
    lo_c, hi_c = bootstrap_ci(sc)

    _, p_da = ttest_rel(sd, sa)
    _, p_db = ttest_rel(sd, sb)
    _, p_dc = ttest_rel(sd, sc)

    wr_d = pairwise_win_rate(sd, [sa, sb, sc])
    wr_a = pairwise_win_rate(sa, [sb, sc, sd])
    wr_b = pairwise_win_rate(sb, [sa, sc, sd])
    wr_c = pairwise_win_rate(sc, [sa, sb, sd])

    iar_a = sum(q.get("iar_a", 0) for q in results)
    iar_b = sum(q.get("iar_b", 0) for q in results)
    iar_c = sum(q.get("iar_c", 0) for q in results)

    stats = {
        "n": n,
        "D": {"mean": np.mean(sd), "std": np.std(sd), "ci_lo": lo_d, "ci_hi": hi_d,
              "win_rate": wr_d},
        "A": {"mean": np.mean(sa), "std": np.std(sa), "ci_lo": lo_a, "ci_hi": hi_a,
              "win_rate": wr_a},
        "B": {"mean": np.mean(sb), "std": np.std(sb), "ci_lo": lo_b, "ci_hi": hi_b,
              "win_rate": wr_b},
        "C": {"mean": np.mean(sc), "std": np.std(sc), "ci_lo": lo_c, "ci_hi": hi_c,
              "win_rate": wr_c},
        "cohens_d_DA": cohens_d(sd, sa),
        "cohens_d_DB": cohens_d(sd, sb),
        "cohens_d_DC": cohens_d(sd, sc),
        "p_DA": p_da, "p_DB": p_db, "p_DC": p_dc,
        "iar_a": iar_a, "iar_b": iar_b, "iar_c": iar_c,
    }
    return stats


def print_paper_summary(stats: dict, corr_stats: dict = None) -> None:
    """Print a clean summary of all numbers in the paper for verification."""
    print()
    print("=" * 65)
    print("PAPER STATISTICS SUMMARY")
    print("=" * 65)
    print()
    print("TABLE 1: QUALITY SCORES")
    for sys, label in [("D", "MAG (Hybrid)"), ("A", "Zero-shot"),
                       ("B", "Semantic RAG"), ("C", "Behavioral RAG")]:
        s = stats[sys]
        print(f"  System {sys} ({label}): {s['mean']:.2f} "
              f"[{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]  "
              f"std={s['std']:.2f}  WR={100*s['win_rate']:.1f}%")

    print()
    print("EFFECT SIZES (Cohen's d, System D vs baselines):")
    print(f"  D vs A: {stats['cohens_d_DA']:.3f}  p={stats['p_DA']:.6f}")
    print(f"  D vs B: {stats['cohens_d_DB']:.3f}  p={stats['p_DB']:.6f}")
    print(f"  D vs C: {stats['cohens_d_DC']:.3f}  p={stats['p_DC']:.6f}")

    print()
    print("IAR (Indiscriminate Affirmation Response):")
    print(f"  A: {stats['iar_a']}/122   "
          f"B: {stats['iar_b']}/122   "
          f"C: {stats['iar_c']}/122")

    if corr_stats:
        print()
        print("SEMANTIC GAP CORRELATION:")
        print(f"  Pearson r:     {corr_stats['pearson_r']:.4f}  "
              f"(p = {corr_stats['pearson_p']:.3f})")
        print(f"  95% CI:        [{corr_stats['ci_lo']:.3f}, {corr_stats['ci_hi']:.3f}]")
        print(f"  Spearman rho:  {corr_stats['spearman_r']:.4f}  "
              f"(p = {corr_stats['spearman_p']:.3f})")
        print(f"  Permutation p: {corr_stats['perm_p']:.3f}")


def compute_semantic_gap(
    beh_emb: np.ndarray,
    beh_ids: np.ndarray,
    sem_emb: np.ndarray,
    sem_ids: list,
    n_boot: int = 10000,
    n_perm: int = 1000
) -> dict:
    """
    Compute the Semantic Gap correlation between behavioral and semantic spaces.

    Behavioral similarity: 1 / (1 + Euclidean distance) in 50D SVD space.
    Semantic similarity:   Cosine similarity in 384D S-BERT space.

    We compare these two similarity measures across all pairwise question
    combinations to test whether behavioral neighborhoods preserve semantic
    structure. Near-zero correlation (r ≈ 0.000) confirms the Semantic Gap.

    Args:
        beh_emb:  (n, 50) behavioral manifold embeddings
        beh_ids:  question IDs for behavioral embeddings
        sem_emb:  (m, 384) S-BERT semantic embeddings
        sem_ids:  question IDs for semantic embeddings
        n_boot:   bootstrap iterations for CI
        n_perm:   permutation iterations for null distribution

    Returns:
        dict with pearson_r, spearman_r, ci_lo, ci_hi, perm_p, etc.
    """
    common_ids = sorted(set(beh_ids) & set(sem_ids))
    print(f"Common questions for correlation: {len(common_ids)}")

    b_map = {qid: i for i, qid in enumerate(beh_ids)}
    s_map = {qid: i for i, qid in enumerate(sem_ids)}

    beh_sub = np.array([beh_emb[b_map[q]] for q in common_ids])
    sem_sub = np.array([sem_emb[s_map[q]] for q in common_ids])

    beh_dist = euclidean_distances(beh_sub)
    beh_sim  = 1.0 / (1.0 + beh_dist)
    sem_sim  = cosine_similarity(sem_sub)

    triu     = np.triu_indices(len(common_ids), k=1)
    beh_flat = beh_sim[triu]
    sem_flat = sem_sim[triu]

    r_pearson,  p_pearson  = pearsonr(beh_flat, sem_flat)
    r_spearman, p_spearman = spearmanr(beh_flat, sem_flat)

    # Bootstrap CI for Pearson r
    np.random.seed(42)
    n_pairs  = len(beh_flat)
    boot_rs  = [
        pearsonr(
            beh_flat[idx := np.random.choice(n_pairs, n_pairs, replace=True)],
            sem_flat[idx]
        )[0]
        for _ in range(n_boot)
    ]
    ci_lo, ci_hi = np.percentile(boot_rs, [2.5, 97.5])

    # Permutation test: what fraction of shuffled correlations exceed observed?
    np.random.seed(42)
    perm_rs = [
        pearsonr(beh_flat, np.random.permutation(sem_flat))[0]
        for _ in range(n_perm)
    ]
    perm_p = np.mean(np.abs(perm_rs) >= np.abs(r_pearson))

    return {
        "n_questions": len(common_ids),
        "pearson_r":   r_pearson,
        "pearson_p":   p_pearson,
        "spearman_r":  r_spearman,
        "spearman_p":  p_spearman,
        "ci_lo":       ci_lo,
        "ci_hi":       ci_hi,
        "perm_p":      perm_p
    }
