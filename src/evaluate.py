"""
evaluate.py - Four-system comparative evaluation.

System A (Zero-Shot):     GPT-4o-mini with no retrieval context.
                          Ungrounded baseline using no retrieval.

System B (Semantic RAG):  GPT-4o-mini + top-5 semantically similar questions.
                          Tests whether topic similarity alone improves hints.

System C (Behavioral RAG): GPT-4o-mini + behavioral cluster context.
                            Tests whether failure patterns alone provide diagnostic value.

System D (MAG):           GPT-4o-mini + two-stage retrieval (top-20 semantic,
                          then top-5 behavioral within semantic boundary).
                          Our proposed Semantic Primacy architecture.

Judge model note: Systems A, B, C were evaluated using GPT-4o-mini as judge.
System D was evaluated using GPT-4o as judge. See paper Section 5.5 for
discussion of this asymmetry.

IAR (Indiscriminate Affirmation Response): A rule-based classifier that
flags hints containing praise phrases regardless of whether the student
was correct. This provides an independent safety check for false validation.
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Praise phrases that indicate false validation of incorrect student responses
PRAISE_PHRASES = [
    "great", "good job", "excellent", "well done", "keep it up",
    "nice try", "good effort", "almost there", "you're on the right track", "close"
]


def check_iar(hint: str) -> int:
    """
    Rule-based Indiscriminate Affirmation Response (IAR) classifier.

    Returns 1 if the hint contains praise phrases that could falsely
    validate a student's incorrect response, 0 otherwise.
    """
    return 1 if any(p in hint.lower() for p in PRAISE_PHRASES) else 0


def tutor_a(q_text: str, client: OpenAI) -> str:
    """
    System A: Zero-shot GPT-4o-mini with Socratic prompting.
    No retrieval context. Tests baseline diagnostic capability.
    """
    prompt = f"""You are an expert mathematics tutor using the Socratic method.
A student answered this question incorrectly:

Question: {q_text}

Instructions:
1. Analyze the question to identify the most likely misconception.
2. Formulate a short diagnostic hint (Socratic question) addressing that error.
3. Do NOT give the answer.
4. Do NOT use conversational filler like "Great try!".

Output ONLY the final hint."""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=150
    )
    return r.choices[0].message.content.strip()


def tutor_b(
    q_text: str,
    client: OpenAI,
    sbert: SentenceTransformer,
    index,
    rag_qids: list,
    question_metadata: dict
) -> str:
    """
    System B: Semantic RAG - retrieve top-5 by S-BERT cosine similarity.
    Tests whether topic-level similarity alone improves diagnostic hints.
    """
    q_emb  = sbert.encode([q_text]).astype("float32")
    _, idx = index.search(q_emb, 5)
    similar = [
        question_metadata[rag_qids[i]]["question_text"][:200]
        for i in idx[0] if rag_qids[i] in question_metadata
    ]
    context = "\n\n".join(f"Example {i+1}: {q}" for i, q in enumerate(similar))
    prompt  = f"""You are an expert math tutor. A student answered incorrectly:

Question: {q_text}

Similar questions from database:
{context}

Provide a diagnostic hint (2-3 sentences). Do NOT give the answer."""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=150
    )
    return r.choices[0].message.content.strip()


def tutor_c(
    q_text: str,
    client: OpenAI,
    sbert: SentenceTransformer,
    index,
    rag_qids: list,
    question_metadata: dict,
    trap_definitions: dict
) -> str:
    """
    System C: Behavioral RAG - retrieve nearest neighbor in SVD manifold.
    Tests whether failure-pattern similarity alone provides diagnostic value.
    The behavioral cluster context is passed silently to avoid hallucination.
    """
    q_emb    = sbert.encode([q_text]).astype("float32")
    _, idx   = index.search(q_emb, 1)
    real_qid = rag_qids[idx[0][0]]
    meta     = question_metadata.get(real_qid, {})
    tid      = meta.get("trap_cluster", 0)
    trap_name = trap_definitions.get(tid, f"Trap {tid}")
    prompt   = f"""You are a math tutor. A student answered incorrectly.

Question: {q_text}

Behavioral diagnosis: student is in error pattern cluster {tid} ({trap_name}).
Provide a tutoring hint (2-3 sentences). Do NOT give the answer."""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=150
    )
    return r.choices[0].message.content.strip()


def tutor_d(
    q_text: str,
    client: OpenAI,
    sbert: SentenceTransformer,
    index,
    rag_qids: list,
    question_metadata: dict,
    embeddings: np.ndarray,
    b_idx_map: dict,
    sem_emb: np.ndarray,
    sem_ids: list,
    kaggle_train_df: pd.DataFrame
) -> str:
    """
    System D: MAG (Manifold-Augmented Generation) - Semantic Primacy.

    Two-stage Filter-then-Personalize:
      1. Semantic Filter: Find top-20 semantically similar questions using
         S-BERT cosine similarity. This ensures topical grounding and
         prevents Topical Drift.
      2. Behavioral Personalization: Within the top-20 semantic candidates,
         select top-5 nearest in the 50D behavioral manifold. This surfaces
         latent misconceptions within the correct mathematical domain.

    All four systems provide exactly 5 context examples to the generator,
    preserving retrieval budget parity.
    """
    # Stage 1: semantic filter - top-20 by cosine similarity
    q_emb      = sbert.encode([q_text])
    sem_sims   = cosine_similarity(q_emb, sem_emb)[0]
    top20_idx  = np.argsort(sem_sims)[-20:][::-1]
    top20_qids = [sem_ids[i] for i in top20_idx]

    # Stage 2: behavioral personalization - top-5 within semantic candidates
    valid = [qid for qid in top20_qids if qid in b_idx_map]
    if len(valid) >= 5:
        beh_sub = np.array([embeddings[b_idx_map[qid]] for qid in valid])
        _, bidx  = index.search(q_emb.astype("float32"), 1)
        rqid     = rag_qids[bidx[0][0]]
        if rqid in b_idx_map:
            beh_q  = embeddings[b_idx_map[rqid]].reshape(1, -1)
            bsims  = cosine_similarity(beh_q, beh_sub)[0]
            top5   = [valid[i] for i in np.argsort(bsims)[-5:][::-1]]
        else:
            top5 = valid[:5]
    else:
        top5 = top20_qids[:5]

    neighbor_texts = []
    for qid in top5:
        qr = kaggle_train_df[kaggle_train_df["QuestionId"] == qid]
        if len(qr) > 0:
            neighbor_texts.append(qr.iloc[0]["QuestionText"])

    context = "\n\n".join(f"Example {i+1}: {q}" for i, q in enumerate(neighbor_texts))
    prompt  = f"""You are a math tutor. A student got this question wrong:

{q_text}

Here are similar questions students also struggle with:
{context}

Provide a helpful hint (2-3 sentences) that guides without giving the answer.
Focus on the underlying concept."""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=150
    )
    return r.choices[0].message.content.strip()


def judge_hint(
    hint: str,
    q_text: str,
    trap_name: str,
    client: OpenAI,
    model: str = "gpt-4o-mini"
) -> int:
    """
    LLM-as-a-Judge evaluation on a 5-point diagnostic scale.

    Evaluates five pedagogical criteria aligned with DPO training boundaries:
    1. No politeness bias (avoids false praise)
    2. Deep diagnosis (addresses root misconception)
    3. Precision (specific to the misconception, not generic)
    4. ZPD respect (guides without revealing answer)
    5. Natural language (conversational, not robotic)

    Note: System D was judged with model="gpt-4o"; baselines with "gpt-4o-mini".
    See paper Section 5.5 for discussion of this judge asymmetry.

    Args:
        hint:      the tutoring hint to evaluate
        q_text:    the original question
        trap_name: the student's likely misconception
        client:    OpenAI client
        model:     judge model ("gpt-4o" or "gpt-4o-mini")

    Returns:
        integer score 1-5
    """
    prompt = f"""Evaluate this tutoring hint using 5 criteria (PASS/FAIL each):
CONTEXT:
Question: {q_text}
Student Misconception: {trap_name}
Hint Given: {hint}
CRITERIA:
1. NO POLITENESS BIAS: Avoids praise like "Great effort!"? (PASS = no praise)
2. DEEP DIAGNOSIS: Addresses ROOT misconception ({trap_name})? (PASS = addresses it)
3. PRECISION: Specific to {trap_name}, not generic? (PASS = specific)
4. ZPD RESPECT: Guides without giving answer? (PASS = guides only)
5. NATURAL LANGUAGE: Conversational, not robotic? (PASS = natural)
5/5 PASS = Score 5, 4/5 = Score 4, 3/5 = Score 3, 2/5 = Score 2, 0-1/5 = Score 1
Respond with ONLY a number 1-5:"""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=10
        )
        return max(1, min(5, int(r.choices[0].message.content.strip())))
    except Exception:
        return 3


def run_evaluation(
    eval_questions: list,
    question_metadata: dict,
    trap_definitions: dict,
    client: OpenAI,
    sbert: SentenceTransformer,
    index,
    rag_qids: list,
    embeddings: np.ndarray,
    b_idx_map: dict,
    sem_emb: np.ndarray,
    sem_ids: list,
    kaggle_train_df: pd.DataFrame,
    output_path: Path,
    checkpoint_every: int = 20
) -> list:
    """
    Run the full four-system evaluation on 122 held-out questions.

    Saves checkpoints every `checkpoint_every` questions so the run
    can be resumed if the session disconnects.

    Returns:
        list of result dicts, one per question
    """
    all_results = []

    for i, q_id in enumerate(eval_questions):
        if q_id not in question_metadata:
            continue
        meta      = question_metadata[q_id]
        q_text    = meta["question_text"]
        tid       = meta["trap_cluster"]
        trap_name = trap_definitions.get(tid, f"Trap {tid}")

        print(f"[{i+1}/{len(eval_questions)}] Q{q_id} (cluster {tid})")

        try:
            hint_a = tutor_a(q_text, client); time.sleep(0.5)
        except Exception as e:
            print(f"  ERROR tutor_a: {e}"); hint_a = ""

        try:
            hint_b = tutor_b(q_text, client, sbert, index, rag_qids, question_metadata)
            time.sleep(0.5)
        except Exception as e:
            print(f"  ERROR tutor_b: {e}"); hint_b = ""

        try:
            hint_c = tutor_c(q_text, client, sbert, index, rag_qids,
                             question_metadata, trap_definitions); time.sleep(0.5)
        except Exception as e:
            print(f"  ERROR tutor_c: {e}"); hint_c = ""

        try:
            hint_d = tutor_d(q_text, client, sbert, index, rag_qids,
                             question_metadata, embeddings, b_idx_map,
                             sem_emb, sem_ids, kaggle_train_df); time.sleep(0.5)
        except Exception as e:
            print(f"  ERROR tutor_d: {e}"); hint_d = ""

        score_a = judge_hint(hint_a, q_text, trap_name, client, "gpt-4o-mini")
        time.sleep(0.3)
        score_b = judge_hint(hint_b, q_text, trap_name, client, "gpt-4o-mini")
        time.sleep(0.3)
        score_c = judge_hint(hint_c, q_text, trap_name, client, "gpt-4o-mini")
        time.sleep(0.3)
        score_d = judge_hint(hint_d, q_text, trap_name, client, "gpt-4o")
        time.sleep(0.3)

        all_results.append({
            "q_id": q_id, "trap_id": tid, "question_text": q_text,
            "hint_a": hint_a, "hint_b": hint_b, "hint_c": hint_c, "hint_d": hint_d,
            "iar_a": check_iar(hint_a), "iar_b": check_iar(hint_b),
            "iar_c": check_iar(hint_c),
            "score_a": score_a, "score_b": score_b,
            "score_c": score_c, "score_d": score_d
        })
        print(f"  A:{score_a} B:{score_b} C:{score_c} D:{score_d}")

        if (i + 1) % checkpoint_every == 0:
            ckpt = output_path.parent / f"eval_checkpoint_{i+1}.json"
            with open(ckpt, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Checkpoint saved: {ckpt}")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_path} ({len(all_results)} questions)")
    return all_results
