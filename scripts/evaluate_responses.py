import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

from metrics import RegressionMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate chatbot responses from output file and generate metrics."
    )
    parser.add_argument(
        "--input",
        # put the last used input file path here as the default
        default="outputs/chatbot_responses_20260412_183114.txt",
        help="Path to the chatbot responses output file.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for manual scoring of each response.",
    )
    parser.add_argument(
        "--auto-classification",
        action="store_true",
        help="Automatically determine classification metrics from response content.",
    )
    parser.add_argument(
        "--auto-regression",
        action="store_true",
        help="Automatically determine regression scores from response content.",
    )
    return parser.parse_args()


def parse_responses(file_path: Path) -> List[Tuple[int, str, str]]:
    """Parse the output file to extract question-answer pairs."""
    if not file_path.exists():
        raise FileNotFoundError(f"Response file not found: {file_path}")
    
    content = file_path.read_text(encoding="utf-8")
    
    # Split by question sections
    sections = re.split(r'-{80}', content)
    
    responses = []
    for section in sections:
        # Extract question and answer
        question_match = re.search(r'Question (\d+): (.+)', section)
        # Capture full answer including citations and evidence
        answer_match = re.search(r'Answer \d+: (.+)', section, re.DOTALL)
        
        if question_match and answer_match:
            q_id = int(question_match.group(1))
            question = question_match.group(2).strip()
            answer = answer_match.group(1).strip()
            responses.append((q_id, question, answer))
    
    return responses


def auto_classify_response(question: str, answer: str) -> tuple[bool, bool, bool, bool | None]:
    """
    Classification metrics for updated response format:
    - answer_correctness: Pass if the answer provides a relevant, specific, and non-generic answer to the question.
    - citation_presence: Pass if the answer includes at least one explicit citation (URL, doc link, or citation block).
    - relevant_retrieval: Pass if the evidence/citations shown are relevant to the answer (matched terms, snippets, or titles overlap with question/answer).
    - correct_refusal: 
        - Pass if the system explicitly declines to answer due to lack of knowledge/evidence.
        - N/A if a substantive answer is provided.
        - Fail only when output is missing/empty and there is no refusal.
    """
    answer_lc = answer.lower()
    question_lc = question.lower()

    refusal_patterns = [
        "[error]", "cannot answer", "don't know", "no information", "not supported", "couldn't generate",
        "out of scope", "i do not have", "i'm unable", "i am unable", "i cannot", "i'm not able",
        "i do not know", "low confidence", "no results found", "unable to find", "not available",
        "outside knowledge", "unsupported", "no relevant information", "insufficient data",
        "unable to access", "unable to retrieve", "encountered an issue retrieving", "technical issue"
    ]
    explicit_refusal = any(pat in answer_lc for pat in refusal_patterns)

    # Substantive answer: long enough and not just a refusal
    answer_present = len(answer.strip()) > 50
    keywords = [w for w in re.findall(r'\w+', question_lc) if w not in {
        "what", "does", "do", "is", "the", "and", "of", "to", "in", "for",
        "how", "can", "with", "or", "after", "a", "an", "on", "by", "from",
        "why", "are", "as", "at", "it", "that", "this", "be", "who", "when",
        "where", "which", "we", "our"
    }]
    keyword_match = any(kw in answer_lc for kw in keywords)

    # Substantive content: answer is present and has keyword overlap
    substantive_content = answer_present and keyword_match

    # Answer correctness: relevant answer that is not just a refusal
    answer_correctness = not explicit_refusal and keyword_match and answer_present

    # Citation presence: URL, doc link, citation block, or email
    citation_patterns = [
        r'https?://\S+',
        r'\[.*?\]\(https?://.*?\)',
        r'\bdoi:\s*\S+',
        r'\bsource[s]?:',
        r'\|.*https?://.*\|',
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        r'citations?:',  # block label
        r'evidence:',    # block label
    ]
    citation_presence = any(re.search(pat, answer, re.IGNORECASE) for pat in citation_patterns)

    # Relevant retrieval: evidence/citation block with keyword overlap or matched/snippet
    relevant_retrieval = False
    if re.search(r'(evidence:|citations?:)', answer, re.IGNORECASE):
        if re.search(r'matched:|snippet:|rank \d+ \|', answer, re.IGNORECASE):
            relevant_retrieval = True
        else:
            blocks = re.split(r'(citations?:|evidence:)', answer, flags=re.IGNORECASE)
            if len(blocks) > 2:
                block_text = " ".join(blocks[2:]).lower()
                keyword_hits = sum(1 for kw in keywords if kw in block_text)
                if keyword_hits >= 2:
                    relevant_retrieval = True
    if not relevant_retrieval:
        keyword_hits = sum(1 for kw in keywords if kw in answer_lc)
        if keyword_hits >= 2 and citation_presence:
            relevant_retrieval = True

    # Correct refusal logic:
    # - Pass if explicit refusal (regardless of answer_present)
    # - N/A if substantive answer is present (long enough and keywords)
    # - Fail only if answer is missing/empty and there is no refusal indication
    if explicit_refusal:
        correct_refusal = True
    elif substantive_content:
        correct_refusal = None
    elif len(answer.strip()) == 0:
        # Only fail if answer is empty and there is no explicit refusal
        correct_refusal = False
    else:
        # Not a refusal, not substantive, but not empty (e.g., short/generic)
        correct_refusal = None

    return answer_correctness, citation_presence, relevant_retrieval, correct_refusal


def auto_score_regression(question: str, answer: str) -> Tuple[int, int, int]:
    """Automatically determine regression scores from response content."""
    answer_lc = answer.lower()
    question_lc = question.lower()

    answer_quality = 3
    grounding_quality = 3
    helpfulness = 3

    technical_issue_patterns = [
        "encountered an issue retrieving", "unable to access", "unable to retrieve",
        "technical issue", "currently cannot access", "am currently unable to access",
        "could not find", "no information available"
    ]
    has_technical_issue = any(pat in answer_lc for pat in technical_issue_patterns)
    has_substantive_content = len(answer.strip()) > 100 and "however" in answer_lc

    if answer.startswith("[ERROR]"):
        answer_quality = 1
        grounding_quality = 1
        helpfulness = 1
    elif has_technical_issue and not has_substantive_content:
        answer_quality = 1
        grounding_quality = 1
        helpfulness = 1
    elif has_technical_issue and has_substantive_content:
        answer_quality = 3
        grounding_quality = 2
        helpfulness = 4
    else:
        word_count = len(answer.split())
        if word_count < 30:
            answer_quality = max(1, answer_quality - 2)
            helpfulness = max(1, helpfulness - 2)
        elif word_count > 100:
            answer_quality = min(5, answer_quality + 1)
            helpfulness = min(5, helpfulness + 1)

        if re.search(r'http[s]?://', answer):
            grounding_quality = min(5, grounding_quality + 2)
        elif '|' in answer:
            grounding_quality = min(5, grounding_quality + 1)
        else:
            grounding_quality = max(1, grounding_quality - 1)

        if re.search(r'\d+\.', answer):
            answer_quality = min(5, answer_quality + 1)
            helpfulness = min(5, helpfulness + 1)

        if any(term in answer_lc for term in ['comprehensive', 'detailed', 'evidence', 'research', 'framework', 'model']):
            answer_quality = min(5, answer_quality + 1)
            grounding_quality = min(5, grounding_quality + 1)

        if any(term in answer_lc for term in ['contact', 'email', 'website', 'learn more']):
            helpfulness = min(5, helpfulness + 1)

    return answer_quality, grounding_quality, helpfulness


def get_manual_scores(question: str, answer: str) -> Tuple[int, int, int, bool, bool, bool, bool]:
    """Prompt user for manual scoring of a response."""
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print(f"Answer: {answer[:500]}{'...' if len(answer) > 500 else ''}")
    print(f"{'='*80}")
    
    # Regression scores
    answer_quality = int(input("Answer Quality (1-5): "))
    grounding_quality = int(input("Grounding Quality (1-5): "))
    helpfulness = int(input("Helpfulness (1-5): "))
    
    # Classification scores
    answer_correctness = input("Answer Correctness (y/n): ").lower().startswith('y')
    citation_presence = input("Citation Presence (y/n): ").lower().startswith('y')
    relevant_retrieval = input("Relevant Retrieval (y/n): ").lower().startswith('y')
    correct_refusal = input("Correct Refusal (y/n): ").lower().startswith('y')
    
    return answer_quality, grounding_quality, helpfulness, answer_correctness, citation_presence, relevant_retrieval, correct_refusal


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent.parent / input_path
    
    # Parse responses
    responses = parse_responses(input_path)
    if not responses:
        print(f"No responses found in {input_path}")
        return
    
    print(f"Found {len(responses)} responses to evaluate")
    
    metrics = RegressionMetrics()
    
    # Evaluate each response
    for q_id, question, answer in responses:
        if args.interactive:
            # Manual scoring
            scores = get_manual_scores(question, answer)
            answer_quality, grounding_quality, helpfulness, answer_correctness, citation_presence, relevant_retrieval, correct_refusal = scores
            notes = "Manual evaluation"
        elif args.auto_classification and args.auto_regression:
            # Auto classification and regression
            answer_correctness, citation_presence, relevant_retrieval, correct_refusal = auto_classify_response(question, answer)
            answer_quality, grounding_quality, helpfulness = auto_score_regression(question, answer)
            notes = ""
        elif args.auto_classification:
            # Auto classification, placeholder regression scores
            answer_correctness, citation_presence, relevant_retrieval, correct_refusal = auto_classify_response(question, answer)
            # For regression, use default scores (can be adjusted)
            answer_quality = 4 if answer_correctness else 2
            grounding_quality = 4 if citation_presence else 2
            helpfulness = 4 if relevant_retrieval else 2
            notes = "Auto-classification with default regression scores"
        else:
            # Skip evaluation
            continue
        
        # Add to metrics
        metrics.add_evaluation(
            question_id=q_id,
            question_text=question,
            answer_text=answer,
            answer_quality=answer_quality,
            grounding_quality=grounding_quality,
            helpfulness=helpfulness,
            answer_correctness=answer_correctness,
            citation_presence=citation_presence,
            relevant_retrieval=relevant_retrieval,
            correct_refusal=correct_refusal,  # <-- do not convert None to False
            evaluator_notes=notes
        )
    
    # Display results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    combined = metrics.get_combined_metrics()
    
    # Regression metrics
    print("\n📊 REGRESSION METRICS:")
    print("-" * 80)
    regression = combined["regression"]
    for key, value in regression.items():
        if key != "overall_distribution":
            print(f"  {key}: {value}")
    print(f"  overall_distribution: {regression['overall_distribution']}")
    
    # Classification metrics
    print("\n" + "=" * 80)
    print("📋 CLASSIFICATION METRICS:")
    print("=" * 80)
    classification = combined["classification"]
    for metric, stats in classification.items():
        if metric != "total_responses_evaluated":
            print(f"\n{metric.upper().replace('_', ' ')}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
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
        # Fix: Show 'N/A' for correct_refusal if it is None, otherwise Pass/Fail
        if class_summary['correct_refusal'] is None:
            refusal_status = 'N/A'
        elif class_summary['correct_refusal'] is True:
            refusal_status = 'Pass'
        else:
            refusal_status = 'Fail'
        print(
            f"     Classification: Correct={'Pass' if class_summary['answer_correctness'] else 'Fail'}, "
            f"Citations={'Pass' if class_summary['citation_presence'] else 'Fail'}, "
            f"Retrieval={'Pass' if class_summary['relevant_retrieval'] else 'Fail'}, "
            f"Refusal={refusal_status}"
        )
        if summary['evaluator_notes']:
            print(f"     Notes: {summary['evaluator_notes']}")

if __name__ == "__main__":
    main()
