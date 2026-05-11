from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from statistics import mean, stdev
from datetime import datetime
import re


MATCH_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "as", "at", "from", "that",
    "this", "it", "about", "what", "who", "when", "where", "why", "how",
    "does", "do", "did", "can", "could", "should", "would", "i", "we",
    "you", "they", "them", "their", "our", "your", "into", "than", "then",
    "have", "has", "had", "will", "want", "need",
}
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")
BENCHMARK_PLACEHOLDER_VALUES = {
    "",
    "[path/to/answer]",
    "path/to/answer",
    "n/a",
    "na",
    "none",
    "null",
    "tbd",
}

@dataclass
class ResponseEvaluation:
    """Stores evaluation scores for a single chatbot response."""
    question_id: int
    question_text: str
    answer_text: str
    # Regression metrics (1-5 scale)
    answer_quality: int  # 1-5: accuracy and completeness
    grounding_quality: int  # 1-5: support by retrieved evidence
    helpfulness: int  # 1-5: usefulness to real user
    # Classification metrics (Pass/Fail)
    answer_correctness: bool  # Pass/Fail: matches reference answer
    citation_presence: bool  # Pass/Fail: includes explicit citations
    relevant_retrieval: bool  # Pass/Fail: retrieved passages support answer
    correct_refusal: bool | None  # Pass/Fail/N/A for proper refusal
    evaluator_notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_average_score(self) -> float:
        """Calculate average score across regression dimensions."""
        return round((self.answer_quality + self.grounding_quality + self.helpfulness) / 3, 2)
    
    def get_classification_summary(self) -> Dict[str, bool | None]:
        """Get summary of classification metrics."""
        return {
            "answer_correctness": self.answer_correctness,
            "citation_presence": self.citation_presence,
            "relevant_retrieval": self.relevant_retrieval,
            "correct_refusal": self.correct_refusal
        }


def _normalize_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _tokenize_for_match(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_PATTERN.findall(_normalize_text(text)):
        token = raw.strip("'")
        if token and token not in MATCH_STOPWORDS and len(token) > 1:
            tokens.append(token)
    return tokens


def _is_placeholder_value(value: str) -> bool:
    normalized = _normalize_text(value)
    return normalized in BENCHMARK_PLACEHOLDER_VALUES


def _result_haystack(result: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(result.get("title", "")),
            str(result.get("url", "")),
            str(result.get("document_id", "")),
            str(result.get("chunk_id", "")),
            str(result.get("evidence_snippet", "")),
            str(result.get("excerpt", "")),
            str(result.get("text", "")),
            str((result.get("citation") or {}).get("title", "")),
            str((result.get("citation") or {}).get("url", "")),
        ]
    )


def _source_doc_aliases(source_doc: str) -> list[str]:
    normalized = _normalize_text(source_doc)
    if not normalized:
        return []

    aliases = {normalized}
    split_tokens = [token for token in re.split(r"[/\\?#]", normalized) if token]
    if split_tokens:
        aliases.add(split_tokens[-1])

    flattened = normalized.replace("/", " ").replace("\\", " ")
    aliases.add(re.sub(r"\s+", " ", flattened).strip())
    return [alias for alias in aliases if alias and not _is_placeholder_value(alias)]


def _text_overlap_ratio(reference_text: str, candidate_text: str) -> float:
    ref_tokens = _tokenize_for_match(reference_text)
    candidate_tokens = set(_tokenize_for_match(candidate_text))
    if not ref_tokens:
        return 0.0
    overlap = sum(1 for token in ref_tokens if token in candidate_tokens)
    return overlap / len(ref_tokens)


def _text_match(reference_text: str, candidate_text: str, threshold: float = 0.65) -> bool:
    ref_norm = _normalize_text(reference_text)
    candidate_norm = _normalize_text(candidate_text)
    if not ref_norm or not candidate_norm:
        return False

    if ref_norm in candidate_norm and len(ref_norm) >= 8:
        return True

    return _text_overlap_ratio(reference_text, candidate_text) >= threshold


def _doc_matches_source(doc_candidate: str, source_doc_aliases: list[str]) -> bool:
    normalized_candidate = _normalize_text(doc_candidate)
    if not normalized_candidate:
        return False

    for alias in source_doc_aliases:
        if alias and (alias in normalized_candidate or normalized_candidate in alias):
            return True
    return False


def _collect_doc_candidates(retrieved_entries: list[Dict[str, Any]]) -> list[str]:
    candidates: list[str] = []
    for entry in retrieved_entries:
        raw_citation = entry.get("citation")
        citation: Dict[str, Any] = raw_citation if isinstance(raw_citation, dict) else {}
        candidates.extend(
            [
                str(entry.get("document_id", "")),
                str(entry.get("chunk_id", "")),
                str(entry.get("url", "")),
                str(entry.get("title", "")),
                str(citation.get("url", "")),
                str(citation.get("title", "")),
            ]
        )
    return [candidate for candidate in candidates if _normalize_text(candidate)]


def _collect_retrieved_corpus_text(retrieved_entries: list[Dict[str, Any]]) -> str:
    return " ".join(_result_haystack(entry) for entry in retrieved_entries)


def _analysis_likely_failure_mode(benchmark_item: Dict[str, Any]) -> str:
    return str(
        benchmark_item.get("phase1_likely_failure_mode")
        or benchmark_item.get("phase2_likely_failure_mode")
        or benchmark_item.get("likely_failure_mode")
        or ""
    ).strip()


def _deterministic_score(correct: int, retrieval_hit: int, grounding_verified: int) -> int:
    # 1-5 score derived only from required core signals.
    return max(1, min(5, 1 + (2 * correct) + retrieval_hit + grounding_verified))


def score_benchmark_item(
    benchmark_item: Dict[str, Any],
    candidate_text: str,
    retrieved_entries: list[Dict[str, Any]] | None = None,
    retrieval_status: str = "ok",
    error_code: str | None = None,
) -> Dict[str, Any]:
    """
    Deterministic benchmark scoring for both retrieval-only and final-answer evaluation.

    Scoring inputs:
    - correctness from ground_truth
    - retrieval_hit from source_doc
    - optional grounding verification from source_text

    Analysis-only inputs (never used for score):
    - why_hard
    - phase*_likely_failure_mode
    - expected_source_hints
    """
    retrieved_entries = retrieved_entries or []
    question = str(benchmark_item.get("question", "")).strip()
    evaluation_text = str(candidate_text or "").strip()
    retrieved_corpus = _collect_retrieved_corpus_text(retrieved_entries)
    if not evaluation_text:
        evaluation_text = retrieved_corpus

    ground_truth = str(benchmark_item.get("ground_truth", "") or "").strip()
    source_doc = str(benchmark_item.get("source_doc", "") or "").strip()
    source_text = str(benchmark_item.get("source_text", "") or "").strip()

    if _is_placeholder_value(ground_truth):
        ground_truth = ""
    if _is_placeholder_value(source_doc):
        source_doc = ""
    if _is_placeholder_value(source_text):
        source_text = ""

    correct = int(bool(ground_truth) and _text_match(ground_truth, evaluation_text, threshold=0.65))

    source_aliases = _source_doc_aliases(source_doc)
    doc_candidates = _collect_doc_candidates(retrieved_entries)
    retrieval_hit = int(bool(source_aliases) and any(_doc_matches_source(candidate, source_aliases) for candidate in doc_candidates))

    grounding_verified = int(bool(source_text) and _text_match(source_text, evaluation_text, threshold=0.50))

    hints = [str(h).strip().lower() for h in (benchmark_item.get("expected_source_hints", []) or []) if str(h).strip()]
    retrieved_lc = _normalize_text(retrieved_corpus)
    hint_match = int(bool(hints) and any(hint in retrieved_lc for hint in hints))

    score = _deterministic_score(correct=correct, retrieval_hit=retrieval_hit, grounding_verified=grounding_verified)

    analysis_bits: list[str] = []
    why_hard = str(benchmark_item.get("why_hard", "") or "").strip()
    likely_failure_mode = _analysis_likely_failure_mode(benchmark_item)
    if why_hard:
        analysis_bits.append(f"why_hard={why_hard}")
    if likely_failure_mode:
        analysis_bits.append(f"likely_failure_mode={likely_failure_mode}")
    if hints and not hint_match:
        analysis_bits.append("hint_miss")
    if not ground_truth:
        analysis_bits.append("missing_ground_truth")
    if not source_doc:
        analysis_bits.append("missing_source_doc")

    notes = " | ".join(analysis_bits)

    return {
        "question": question,
        "correct": correct,
        "retrieval_hit": retrieval_hit,
        "score": score,
        "hint_match": hint_match,
        "notes": notes,
        "retrieval_status": retrieval_status,
        "error_code": error_code,
        "grounding_verified": grounding_verified,
    }


def score_retrieval_benchmark_item(
    benchmark_item: Dict[str, Any],
    retrieval_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Adapter for retrieval benchmark runs (HPC + local variant evaluations)."""
    ranked_results = retrieval_result.get("results", []) if isinstance(retrieval_result, dict) else []
    retrieval_status = retrieval_result.get("retrieval_status", "failed") if isinstance(retrieval_result, dict) else "failed"
    error_code = retrieval_result.get("error_code") if isinstance(retrieval_result, dict) else "unknown_error"

    candidate_text = _collect_retrieved_corpus_text(ranked_results)
    payload = score_benchmark_item(
        benchmark_item=benchmark_item,
        candidate_text=candidate_text,
        retrieved_entries=ranked_results,
        retrieval_status=retrieval_status,
        error_code=error_code,
    )
    payload["result_count"] = int(retrieval_result.get("result_count", 0)) if isinstance(retrieval_result, dict) else 0
    return payload


def score_final_answer_benchmark_item(
    benchmark_item: Dict[str, Any],
    answer_text: str,
    retrieved_entries: list[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Adapter for final-response evaluation pipelines that score answer text against benchmark truth."""
    return score_benchmark_item(
        benchmark_item=benchmark_item,
        candidate_text=answer_text,
        retrieved_entries=retrieved_entries,
        retrieval_status="ok",
        error_code=None,
    )


def aggregate_retrieval_benchmark_scores(details: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-item benchmark scores for fair variant comparison."""
    if not details:
        return {
            "accuracy": 0.0,
            "retrieval_hit_rate": 0.0,
            "avg_score": 0.0,
            "hint_match_rate": 0.0,
            "question_count": 0,
            "details": [],
            # Compatibility aliases for older artifacts/consumers.
            "retrieval_score": 0.0,
            "retrieval_failures": 0,
            "failure_rate": 0.0,
        }

    question_count = len(details)
    correct_total = sum(int(d.get("correct", 0)) for d in details)
    retrieval_hit_total = sum(int(d.get("retrieval_hit", 0)) for d in details)
    score_total = sum(int(d.get("score", 1)) for d in details)
    hint_match_total = sum(int(d.get("hint_match", 0)) for d in details)
    retrieval_failures = sum(1 for d in details if d.get("retrieval_status") != "ok")

    accuracy = correct_total / question_count
    retrieval_hit_rate = retrieval_hit_total / question_count
    avg_score = score_total / question_count
    hint_match_rate = hint_match_total / question_count
    failure_rate = retrieval_failures / question_count

    # Normalized 0..1 score alias for older consumers; canonical metric is avg_score (1..5).
    retrieval_score = avg_score / 5.0

    return {
        "accuracy": round(accuracy, 4),
        "retrieval_hit_rate": round(retrieval_hit_rate, 4),
        "avg_score": round(avg_score, 4),
        "hint_match_rate": round(hint_match_rate, 4),
        "question_count": question_count,
        "details": details,
        # Compatibility aliases for existing downstream readers.
        "retrieval_score": round(retrieval_score, 4),
        "retrieval_failures": retrieval_failures,
        "failure_rate": round(failure_rate, 4),
    }


class RegressionMetrics:
    """Calculate regression metrics for continuous 1-5 quality scores."""
    
    def __init__(self):
        self.evaluations: List[ResponseEvaluation] = []
    
    def add_evaluation(
        self,
        question_id: int,
        question_text: str,
        answer_text: str,
        answer_quality: int,
        grounding_quality: int,
        helpfulness: int,
        answer_correctness: bool,
        citation_presence: bool,
        relevant_retrieval: bool,
        correct_refusal: bool | None,
        evaluator_notes: str = ""
    ) -> None:
        """
        Record an evaluation for a chatbot response.
        
        Args:
            question_id: Sequential question number
            question_text: The question asked
            answer_text: The chatbot's answer
            answer_quality: 1-5 score for accuracy and completeness
            grounding_quality: 1-5 score for evidence support
            helpfulness: 1-5 score for user utility
            answer_correctness: Pass/Fail for matching reference answer
            citation_presence: Pass/Fail for including citations
            relevant_retrieval: Pass/Fail for supporting retrieval
            correct_refusal: Pass/Fail for proper refusal
            evaluator_notes: Optional notes from evaluator
        """
        if not all(1 <= score <= 5 for score in [answer_quality, grounding_quality, helpfulness]):
            raise ValueError("All regression scores must be between 1 and 5")
        
        evaluation = ResponseEvaluation(
            question_id=question_id,
            question_text=question_text,
            answer_text=answer_text,
            answer_quality=answer_quality,
            grounding_quality=grounding_quality,
            helpfulness=helpfulness,
            answer_correctness=answer_correctness,
            citation_presence=citation_presence,
            relevant_retrieval=relevant_retrieval,
            correct_refusal=correct_refusal,
            evaluator_notes=evaluator_notes
        )
        self.evaluations.append(evaluation)
    
    def _get_dimension_scores(self, dimension: str) -> List[int]:
        """Extract scores for a specific dimension."""
        if dimension == "answer_quality":
            return [e.answer_quality for e in self.evaluations]
        elif dimension == "grounding_quality":
            return [e.grounding_quality for e in self.evaluations]
        elif dimension == "helpfulness":
            return [e.helpfulness for e in self.evaluations]
        else:
            raise ValueError(f"Unknown dimension: {dimension}")
    
    def _calculate_stats(self, scores: List[int]) -> Dict[str, Any]:
        """Calculate statistical metrics for a list of scores."""
        if not scores:
            return {}
        
        avg = round(mean(scores), 2)
        min_score = min(scores)
        max_score = max(scores)
        
        # Standard deviation (requires at least 2 values)
        std_dev = round(stdev(scores), 2) if len(scores) > 1 else 0.0
        
        # Distribution across 1-5 scale
        distribution = {i: scores.count(i) for i in range(1, 6)}
        
        return {
            "mean": avg,
            "std_dev": std_dev,
            "min": min_score,
            "max": max_score,
            "distribution": distribution,
            "count": len(scores)
        }
    
    def get_dimension_metrics(self, dimension: str) -> Dict[str, Any]:
        """
        Get regression metrics for a specific dimension.
        
        Args:
            dimension: One of 'answer_quality', 'grounding_quality', 'helpfulness'
            
        Returns:
            Dictionary with statistical metrics
        """
        scores = self._get_dimension_scores(dimension)
        stats = self._calculate_stats(scores)
        return {
            "dimension": dimension,
            **stats
        }
    
    def get_all_dimensions_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get regression metrics for all three dimensions."""
        return {
            "answer_quality": self.get_dimension_metrics("answer_quality"),
            "grounding_quality": self.get_dimension_metrics("grounding_quality"),
            "helpfulness": self.get_dimension_metrics("helpfulness")
        }
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """
        Calculate overall performance metrics across all dimensions and responses.
        
        Returns:
            Dictionary with comprehensive metrics
        """
        if not self.evaluations:
            return {}
        
        all_scores = []
        for eval in self.evaluations:
            all_scores.extend([eval.answer_quality, eval.grounding_quality, eval.helpfulness])
        
        stats = self._calculate_stats(all_scores)
        
        return {
            "total_responses_evaluated": len(self.evaluations),
            "total_dimension_scores": len(all_scores),
            "overall_mean": stats["mean"],
            "overall_std_dev": stats["std_dev"],
            "overall_min": stats["min"],
            "overall_max": stats["max"],
            "overall_distribution": stats["distribution"]
        }
    
    def get_response_summary(self, question_id: int) -> Dict[str, Any]:
        """Get detailed evaluation summary for a specific response."""
        for eval in self.evaluations:
            if eval.question_id == question_id:
                return {
                    "question_id": eval.question_id,
                    "question": eval.question_text,
                    "answer_quality": eval.answer_quality,
                    "grounding_quality": eval.grounding_quality,
                    "helpfulness": eval.helpfulness,
                    "average_score": eval.get_average_score(),
                    "evaluator_notes": eval.evaluator_notes,
                    "timestamp": eval.timestamp
                }
        return {}
    
    def get_all_evaluations_summary(self) -> List[Dict[str, Any]]:
        """Get summary for all evaluations."""
        return [
            {
                "question_id": eval.question_id,
                "question": eval.question_text,
                "answer_quality": eval.answer_quality,
                "grounding_quality": eval.grounding_quality,
                "helpfulness": eval.helpfulness,
                "average_score": eval.get_average_score(),
                "correct_refusal": eval.correct_refusal,
                "evaluator_notes": eval.evaluator_notes
            }
            for eval in self.evaluations
        ]
    
    def get_low_performing_responses(self, threshold: float = 3.0) -> List[Dict[str, Any]]:
        """Identify responses scoring below threshold on average."""
        low_performers = []
        for eval in self.evaluations:
            avg = eval.get_average_score()
            if avg <= threshold:
                low_performers.append({
                    "question_id": eval.question_id,
                    "question": eval.question_text,
                    "average_score": avg,
                    "answer_quality": eval.answer_quality,
                    "grounding_quality": eval.grounding_quality,
                    "helpfulness": eval.helpfulness
                })
        return sorted(low_performers, key=lambda x: x["average_score"])
    
    def export_evaluations(self) -> List[Dict[str, Any]]:
        """Export all evaluations as list of dictionaries."""
        return [eval.to_dict() for eval in self.evaluations]
    
    def reset(self) -> None:
        """Clear all evaluations."""
        self.evaluations = []
    
    def get_classification_metrics(self) -> Dict[str, Any]:
        """
        Calculate classification metrics (Pass/Fail rates).
        
        Returns:
            Dictionary with accuracy rates for each classification metric
        """
        if not self.evaluations:
            return {}
        
        total = len(self.evaluations)
        answer_correctness_pass = sum(1 for e in self.evaluations if e.answer_correctness)
        citation_presence_pass = sum(1 for e in self.evaluations if e.citation_presence)
        relevant_retrieval_pass = sum(1 for e in self.evaluations if e.relevant_retrieval)
        
        correct_refusal_applicable = [e for e in self.evaluations if e.correct_refusal is not None]
        correct_refusal_pass = sum(1 for e in correct_refusal_applicable if e.correct_refusal)
        correct_refusal_fail = sum(1 for e in correct_refusal_applicable if e.correct_refusal is False)
        applicable_count = len(correct_refusal_applicable)

        return {
            "total_responses_evaluated": total,
            "answer_correctness": {
                "pass_rate": round(answer_correctness_pass / total, 2),
                "pass_count": answer_correctness_pass,
                "fail_count": total - answer_correctness_pass
            },
            "citation_presence": {
                "pass_rate": round(citation_presence_pass / total, 2),
                "pass_count": citation_presence_pass,
                "fail_count": total - citation_presence_pass
            },
            "relevant_retrieval": {
                "pass_rate": round(relevant_retrieval_pass / total, 2),
                "pass_count": relevant_retrieval_pass,
                "fail_count": total - relevant_retrieval_pass
            },
            "correct_refusal": {
                "pass_rate": round(correct_refusal_pass / applicable_count, 2) if applicable_count else 0.0,
                "pass_count": correct_refusal_pass,
                "fail_count": correct_refusal_fail,
                "not_applicable_count": total - applicable_count
            }
        }
    
    def get_combined_metrics(self) -> Dict[str, Any]:
        """
        Get both regression and classification metrics combined.
        
        Returns:
            Dictionary with all metrics
        """
        return {
            "regression": self.get_overall_metrics(),
            "classification": self.get_classification_metrics()
        }
    
    def get_failed_responses(self, metric: str) -> List[Dict[str, Any]]:
        """Get responses that failed a specific classification metric."""
        failed = []
        for eval in self.evaluations:
            value = getattr(eval, metric)
            if metric == "correct_refusal":
                if value is False:
                    failed.append({
                        "question_id": eval.question_id,
                        "question": eval.question_text,
                        "answer": eval.answer_text,
                        "failed_metric": metric,
                        "regression_avg": eval.get_average_score(),
                        "notes": eval.evaluator_notes
                    })
            elif not value:
                failed.append({
                    "question_id": eval.question_id,
                    "question": eval.question_text,
                    "answer": eval.answer_text,
                    "failed_metric": metric,
                    "regression_avg": eval.get_average_score(),
                    "notes": eval.evaluator_notes
                })
        return failed
    
# Example usage with demo questions
if __name__ == "__main__":
    metrics = RegressionMetrics()
    
    # Sample evaluations based on demo questions
    # Format: (q_id, q_text, answer_q, ground_q, help_q, answer_corr, cit_pres, rel_ret, corr_ref, notes)
    evaluations_data = [
        (1, "What does Early Education Leaders do?", 5, 5, 5, True, True, True, False, "Comprehensive and well-sourced"),
        (2, "What kinds of leadership programs does Early Education Leaders offer?", 4, 5, 4, True, True, True, False, "Good detail, clear structure"),
        (3, "How can someone get involved with Early Education Leaders or learn more?", 5, 4, 5, True, True, True, False, "Actionable contact info provided"),
        (4, "What do graduates actually do after completing Early Education Leaders programs?", 3, 3, 4, True, False, True, False, "Some examples but limited detail"),
        (5, "What kinds of partnerships or collaborations does Early Education Leaders mention?", 5, 5, 5, True, True, True, False, "Well-explained partnerships"),
        (6, "What research does Anne Douglass do?", 4, 4, 4, True, True, True, False, "Good overview of research areas"),
        (7, "What is the ExCELS study and why does it matter?", 1, 1, 1, False, False, False, True, "ERROR: Content empty - correct refusal"),
        (8, "How is Early Education Leaders different from a normal training program?", 5, 5, 5, True, True, True, False, "Clear differentiation points"),
        (9, "What is the Culture of Continuous Learning Project trying to improve?", 4, 4, 4, True, True, True, False, "Well-articulated goals"),
        (10, "What is the Essential Leadership Model?", 4, 4, 4, True, True, True, False, "Good explanation of model")
    ]
    
    for q_id, q_text, answer_q, ground_q, help_q, answer_corr, cit_pres, rel_ret, corr_ref, notes in evaluations_data:
        metrics.add_evaluation(q_id, q_text, "", answer_q, ground_q, help_q, answer_corr, cit_pres, rel_ret, corr_ref, notes)
    
    print("=" * 80)
    print("COMPREHENSIVE METRICS SUMMARY")
    print("=" * 80)
    
    # Regression metrics
    print("\n📊 REGRESSION METRICS:")
    print("-" * 80)
    regression = metrics.get_overall_metrics()
    for key, value in regression.items():
        print(f"  {key}: {value}")
    
    # Classification metrics
    print("\n" + "=" * 80)
    print("📋 CLASSIFICATION METRICS:")
    print("=" * 80)
    classification = metrics.get_classification_metrics()
    for metric, stats in classification.items():
        if metric != "total_responses_evaluated":
            print(f"\n{metric.upper().replace('_', ' ')}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    # Failed responses
    print("\n" + "=" * 80)
    print("❌ FAILED CLASSIFICATION RESPONSES:")
    print("=" * 80)
    for metric in ["answer_correctness", "citation_presence", "relevant_retrieval", "correct_refusal"]:
        failed = metrics.get_failed_responses(metric)
        if failed:
            print(f"\n{metric.upper().replace('_', ' ')} FAILURES:")
            for f in failed:
                print(f"  Q{f['question_id']}: {f['question']}")
    
    # Individual summaries
    print("\n" + "=" * 80)
    print("📋 INDIVIDUAL RESPONSE SUMMARIES:")
    print("=" * 80)
    summaries = metrics.get_all_evaluations_summary()
    for summary in summaries:
        status = "✓" if summary['average_score'] >= 4.0 else "~" if summary['average_score'] >= 3.0 else "✗"
        print(f"\n  {status} Q{summary['question_id']}: {summary['average_score']}/5")
        print(f"     Question: {summary['question']}")
        print(f"     Regression Scores: Quality={summary['answer_quality']}, Grounding={summary['grounding_quality']}, Helpfulness={summary['helpfulness']}")
        # Add classification summary
        eval_obj = next(e for e in metrics.evaluations if e.question_id == summary['question_id'])
        class_summary = eval_obj.get_classification_summary()
        print(f"     Classification: Correct={'Pass' if class_summary['answer_correctness'] else 'Fail'}, Citations={'Pass' if class_summary['citation_presence'] else 'Fail'}, Retrieval={'Pass' if class_summary['relevant_retrieval'] else 'Fail'}, Refusal={'Pass' if class_summary['correct_refusal'] else 'Fail'}")
        if summary['evaluator_notes']:
            print(f"     Notes: {summary['evaluator_notes']}")
