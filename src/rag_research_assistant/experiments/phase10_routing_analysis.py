"""Analyze cheap baseline signals against saved Phase 10B MQ2 outcomes."""

import argparse
import csv
import hashlib
import json
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..routing.signals import (
    BaselineSignals,
    RankedCandidate,
    RoutingExample,
    ThresholdRule,
    evaluate_routing_decisions,
    evaluate_threshold_rule,
    extract_baseline_signals,
    join_mq2_outcomes,
)


DEFAULT_PHASE10B_RESULTS = Path(
    "data/evaluation/benchmarks/phase-10b-multiquery-results.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "data/evaluation/benchmarks/phase-10c-routing-signal-analysis.json"
)
DEFAULT_CSV_OUTPUT = Path(
    "data/evaluation/benchmarks/phase-10c-routing-signals.csv"
)
DEFAULT_MARKDOWN_OUTPUT = Path("docs/phase-10c-routing-signal-analysis.md")


def _candidate(value: Mapping[str, Any]) -> RankedCandidate:
    return RankedCandidate(
        score=float(value["score"]),
        document=str(value["document"]),
        start_page=int(value["start_page"]),
        end_page=int(value["end_page"]),
        chunk_id=str(value["chunk_id"]),
        text=str(value["text_preview"]),
    )


def _stats(values: Sequence[float]) -> Dict[str, Optional[float] | int]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _probability_positive_higher(positive: Sequence[float], negative: Sequence[float]) -> Optional[float]:
    """Mann-Whitney interpretation: P(positive > negative), with ties split."""

    if not positive or not negative:
        return None
    wins = sum(left > right for left in positive for right in negative)
    ties = sum(left == right for left in positive for right in negative)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def _thresholds(values: Sequence[float]) -> List[float]:
    unique = sorted(set(values))
    if not unique:
        return []
    boundaries = [unique[0] - 1.0]
    boundaries.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    boundaries.append(unique[-1] + 1.0)
    return boundaries


def _rule_record(rule: ThresholdRule, metrics) -> Dict[str, Any]:
    return {"rule": asdict(rule), "metrics": asdict(metrics)}


def _best_rule(records: Sequence[Dict[str, Any]], minimum_rescue_recall: float) -> Optional[Dict[str, Any]]:
    eligible = [
        item
        for item in records
        if item["metrics"]["rescue_recall"] >= minimum_rescue_recall
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            item["metrics"]["escalation_rate"],
            -item["metrics"]["routing_precision"],
            item["rule"]["signal"],
            item["rule"]["direction"],
            item["rule"]["threshold"],
        ),
    )


def analyze_phase10b(
    phase10b: Mapping[str, Any], *, source_path: str = ""
) -> Dict[str, Any]:
    questions = phase10b.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Phase 10B report has no questions array")
    scored = [item for item in questions if item.get("answerability") != "unanswerable"]
    if not scored:
        raise ValueError("Phase 10B report has no scored questions")

    signal_by_id: Dict[str, BaselineSignals] = {}
    outcome_by_id: Dict[str, str] = {}
    examples: List[RoutingExample] = []
    per_question: List[Dict[str, Any]] = []
    for item in scored:
        baseline = tuple(_candidate(value) for value in item["baseline_results"])
        hybrid_rankings = item["query_level_candidates"]["multi_query_2"]
        if not hybrid_rankings:
            raise ValueError(f"{item['question_id']}: original hybrid ranking is missing")
        hybrid = tuple(_candidate(value) for value in hybrid_rankings[0])
        signals = extract_baseline_signals(item["original_question"], baseline, hybrid)
        question_id = str(item["question_id"])
        signal_by_id[question_id] = signals
        outcome_by_id[question_id] = str(item["multi_query_2_outcome"])
        example = RoutingExample(
            question_id=question_id,
            mq2_outcome=str(item["multi_query_2_outcome"]),
            baseline_first_relevant_rank=item["baseline_first_relevant_rank"],
            mq2_first_relevant_rank=item["multi_query_2_first_relevant_rank"],
            expected_sources=tuple(item["expected_sources"]),
            baseline_results=tuple(item["baseline_results"]),
            mq2_results=tuple(item["multi_query_2_results"]),
            signals=signals,
        )
        examples.append(example)
        per_question.append(
            {
                "question_id": question_id,
                "original_question": item["original_question"],
                "mq2_outcome": item["multi_query_2_outcome"],
                "baseline_first_relevant_rank": item["baseline_first_relevant_rank"],
                "mq2_first_relevant_rank": item["multi_query_2_first_relevant_rank"],
                "baseline_top_k": [
                    {
                        "rank": value["rank"], "document": value["document"],
                        "start_page": value["start_page"], "end_page": value["end_page"],
                        "chunk_id": value["chunk_id"], "score": value["score"],
                    }
                    for value in item["baseline_results"]
                ],
                "signals": asdict(signals),
            }
        )
    join_mq2_outcomes(signal_by_id, outcome_by_id)

    baseline_latency = float(phase10b["summary"]["average_baseline_retrieval_seconds"])
    mq2_latency = float(phase10b["summary"]["average_multi_query_2_total_seconds"])
    all_signal_names = sorted({name for example in examples for name in example.signals.numeric()})
    candidate_rules: List[Dict[str, Any]] = []
    signal_analysis: Dict[str, Any] = {}
    for name in all_signal_names:
        improved_values = [
            example.signals.numeric()[name]
            for example in examples
            if example.mq2_outcome == "improved" and name in example.signals.numeric()
        ]
        unchanged_values = [
            example.signals.numeric()[name]
            for example in examples
            if example.mq2_outcome == "unchanged" and name in example.signals.numeric()
        ]
        values = improved_values + unchanged_values
        signal_rules = []
        for threshold in _thresholds(values):
            for direction in ("at_or_below", "at_or_above"):
                rule = ThresholdRule(name, direction, threshold)
                record = _rule_record(
                    rule,
                    evaluate_threshold_rule(
                        examples,
                        rule,
                        baseline_latency_seconds=baseline_latency,
                        mq2_latency_seconds=mq2_latency,
                    ),
                )
                signal_rules.append(record)
                candidate_rules.append(record)
        probability = _probability_positive_higher(improved_values, unchanged_values)
        signal_analysis[name] = {
            "mq2_improved": _stats(improved_values),
            "mq2_unchanged": _stats(unchanged_values),
            "probability_improved_value_exceeds_unchanged": probability,
            "separation_strength": (
                None if probability is None else abs(probability - 0.5) * 2.0
            ),
            "direction_suggested_by_rank_statistic": (
                None if probability is None else (
                    "higher_when_helpful" if probability > 0.5 else
                    "lower_when_helpful" if probability < 0.5 else "no_direction"
                )
            ),
            "best_rule_at_100_percent_rescue_recall": _best_rule(signal_rules, 1.0),
            "best_rule_at_80_percent_rescue_recall": _best_rule(signal_rules, 0.8),
        }

    always_baseline = evaluate_routing_decisions(
        examples, [False] * len(examples),
        baseline_latency_seconds=baseline_latency, mq2_latency_seconds=mq2_latency,
    )
    always_mq2 = evaluate_routing_decisions(
        examples, [True] * len(examples),
        baseline_latency_seconds=baseline_latency, mq2_latency_seconds=mq2_latency,
    )
    strongest = sorted(
        (
            {"signal": name, **analysis}
            for name, analysis in signal_analysis.items()
            if analysis["separation_strength"] is not None
        ),
        key=lambda item: (-item["separation_strength"], item["signal"]),
    )
    best_full_rescue = _best_rule(candidate_rules, 1.0)
    best_four_of_five = _best_rule(candidate_rules, 0.8)

    def routed_ids(rule_record: Optional[Dict[str, Any]], *, helpful: bool) -> List[str]:
        if rule_record is None:
            return []
        rule = ThresholdRule(**rule_record["rule"])
        return [
            example.question_id
            for example in examples
            if rule.routes(example.signals)
            and ((example.mq2_outcome == "improved") == helpful)
        ]

    return {
        "schema_version": 1,
        "experiment": "phase-10c-1-baseline-routing-signal-analysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_phase10b_path": source_path,
        "source_phase10b_sha256": None,
        "source_dataset_path": phase10b.get("dataset_path"),
        "source_dataset_sha256": phase10b.get("dataset_sha256"),
        "configuration": {
            "baseline_latency_seconds": baseline_latency,
            "mq2_routed_latency_seconds": mq2_latency,
            "latency_formula": "(1-escalation_rate)*baseline + escalation_rate*mq2_routed",
            "threshold_search": "all observed-value boundaries, both directions; exploratory only",
            "score_calibration_warning": "scores are ranking signals, not probabilities",
            "text_source": "Phase 10B persisted 400-character chunk previews",
        },
        "summary": {
            "total_scored_questions": len(examples),
            "mq2_helpful_questions": sum(example.mq2_outcome == "improved" for example in examples),
            "mq2_unchanged_questions": sum(example.mq2_outcome == "unchanged" for example in examples),
            "mq2_degraded_questions": sum(example.mq2_outcome == "degraded" for example in examples),
            "class_imbalance": "5 helpful vs 85 unchanged; no degraded examples",
            "always_baseline": asdict(always_baseline),
            "always_mq2": asdict(always_mq2),
            "strongest_univariate_signals_by_in_sample_rank_separation": [
                {"signal": item["signal"], "separation_strength": item["separation_strength"],
                 "direction": item["direction_suggested_by_rank_statistic"]}
                for item in strongest[:8]
            ],
            "weakest_univariate_signals_by_in_sample_rank_separation": [
                {"signal": item["signal"], "separation_strength": item["separation_strength"]}
                for item in reversed(strongest[-5:])
            ],
            "lowest_escalation_rule_at_100_percent_rescue_recall": best_full_rescue,
            "lowest_escalation_rule_at_80_percent_rescue_recall": best_four_of_five,
            "representative_helpful_questions": [
                {
                    "question_id": example.question_id,
                    "baseline_first_relevant_rank": example.baseline_first_relevant_rank,
                    "mq2_first_relevant_rank": example.mq2_first_relevant_rank,
                }
                for example in examples
                if example.mq2_outcome == "improved"
            ],
            "representative_false_positives_for_80_percent_rule": routed_ids(
                best_four_of_five, helpful=False
            )[:12],
            "missed_rescues_for_80_percent_rule": [
                example.question_id
                for example in examples
                if example.mq2_outcome == "improved"
                and example.question_id not in routed_ids(best_four_of_five, helpful=True)
            ],
        },
        "unavailable_signals": [
            {
                "signal": "dense_bm25_rank_agreement",
                "reason": "Phase 10B persisted the hybrid RRF ranking, not its separate dense and BM25 component rankings.",
            },
            {
                "signal": "dense_bm25_top_chunk_or_document_agreement",
                "reason": "Separate component rankings would require a new baseline-only retrieval collection run.",
            },
        ],
        "signal_analysis": signal_analysis,
        "candidate_threshold_rules": candidate_rules,
        "per_question": per_question,
        "limitations": [
            "Only five positive cases exist, so all separation and threshold results are unstable.",
            "Thresholds are explored and evaluated on the same benchmark, creating selection leakage and overfitting risk.",
            "Cross-encoder and RRF scores are uncalibrated and must not be interpreted as probabilities.",
            "Query coverage uses persisted 400-character previews rather than complete chunks.",
            "Dense/BM25 component agreement was not available in the saved Phase 10B report.",
        ],
    }


def write_json(report: Dict[str, Any], path: Path, source_path: Path) -> None:
    report["source_phase10b_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(report: Mapping[str, Any], path: Path) -> None:
    rows = report["per_question"]
    signal_names = sorted(rows[0]["signals"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["question_id", "original_question", "mq2_outcome", "baseline_first_relevant_rank", "mq2_first_relevant_rank", *signal_names],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "question_id": row["question_id"], "original_question": row["original_question"],
                "mq2_outcome": row["mq2_outcome"], "baseline_first_relevant_rank": row["baseline_first_relevant_rank"],
                "mq2_first_relevant_rank": row["mq2_first_relevant_rank"], **row["signals"],
            })


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    signals = report["signal_analysis"]
    strongest = summary["strongest_univariate_signals_by_in_sample_rank_separation"]
    lines = [
        "# Phase 10C-1: baseline routing-signal analysis", "",
        "## Scope and labels", "",
        "This offline analysis reuses the saved Phase 10B baseline rankings and MQ2 outcomes. It makes no Ollama calls, does not rerun retrieval, and does not change production routing.", "",
        f"- Scored questions: {summary['total_scored_questions']}",
        f"- MQ2-helpful / unchanged / degraded: {summary['mq2_helpful_questions']} / {summary['mq2_unchanged_questions']} / {summary['mq2_degraded_questions']}",
        f"- Class imbalance: {summary['class_imbalance']}", "",
        "## Cheap signals", "",
        "Signals cover reranker score shape, top-k document concentration, query/protected-term coverage, and hybrid-to-reranked stability. Cross-encoder and RRF scores are treated as relative ranking signals, never probabilities.", "",
        "The separation column is an AUC-style rank statistic rescaled to 0–1: zero means no ordering difference and one means complete ordering in either direction. It measures in-sample ordering, not predictive confidence.", "",
        "Separate dense and BM25 rankings were not persisted by Phase 10B, so dense/BM25 agreement is explicitly unavailable in this pass.", "",
        "## Strongest univariate separation (exploratory)", "",
        "| Signal | In-sample separation | Direction among helpful cases | Best escalation rate for 100% rescue | Precision |", "|---|---:|---|---:|---:|",
    ]
    for item in strongest:
        analysis = signals[item["signal"]]
        best = analysis["best_rule_at_100_percent_rescue_recall"]
        metrics = best["metrics"] if best else None
        lines.append(
            f"| `{item['signal']}` | {item['separation_strength']:.3f} | {item['direction']} | "
            f"{_pct(metrics['escalation_rate']) if metrics else 'n/a'} | "
            f"{_pct(metrics['routing_precision']) if metrics else 'n/a'} |"
        )
    always = summary["always_mq2"]
    full_rescue = summary["lowest_escalation_rule_at_100_percent_rescue_recall"]
    four_rescue = summary["lowest_escalation_rule_at_80_percent_rescue_recall"]
    lines.extend([
        "", "## Routing reference points", "",
        f"Always-baseline has 0% escalation, 0% rescue recall, and estimated average latency {summary['always_baseline']['estimated_average_latency_seconds']:.3f}s.", "",
        f"Always-MQ2 has 100% escalation, 100% rescue recall, {_pct(always['routing_precision'])} routing precision, and estimated average latency {always['estimated_average_latency_seconds']:.3f}s.", "",
        "### Best in-sample threshold trade-offs", "",
        "| Target | Exploratory rule | Escalation | Precision | Missed | Unnecessary | Estimated latency |", "|---|---|---:|---:|---:|---:|---:|",
        f"| Rescue all 5 | `{full_rescue['rule']['signal']} {full_rescue['rule']['direction']} {full_rescue['rule']['threshold']:.4g}` | {_pct(full_rescue['metrics']['escalation_rate'])} | {_pct(full_rescue['metrics']['routing_precision'])} | {full_rescue['metrics']['missed_rescues']} | {full_rescue['metrics']['unnecessary_escalations']} | {full_rescue['metrics']['estimated_average_latency_seconds']:.3f}s |",
        f"| Rescue 4 of 5 | `{four_rescue['rule']['signal']} {four_rescue['rule']['direction']} {four_rescue['rule']['threshold']:.4g}` | {_pct(four_rescue['metrics']['escalation_rate'])} | {_pct(four_rescue['metrics']['routing_precision'])} | {four_rescue['metrics']['missed_rescues']} | {four_rescue['metrics']['unnecessary_escalations']} | {four_rescue['metrics']['estimated_average_latency_seconds']:.3f}s |",
        "",
        f"The four-of-five rule misses `{summary['missed_rescues_for_80_percent_rule'][0]}`. Its first twelve false positives are: "
        + ", ".join(f"`{value}`" for value in summary["representative_false_positives_for_80_percent_rule"])
        + ".", "",
        "Every threshold in the JSON was swept at observed-value boundaries in both directions. These are exploratory in-sample probes, not calibrated or production-ready rules.", "",
        "The score gaps and score spread were among the weakest separators. This is a useful negative result: an apparently confident top cross-encoder score does not reliably identify when MQ2 will help.", "",
        "## Helpful-question examples", "",
    ])
    helpful_by_id = {
        item["question_id"]: item
        for item in report["per_question"]
        if item["mq2_outcome"] == "improved"
    }
    for item in summary["representative_helpful_questions"]:
        question = helpful_by_id[item["question_id"]]["original_question"]
        before = item["baseline_first_relevant_rank"] or "not found"
        lines.append(
            f"- `{item['question_id']}`: {before} → {item['mq2_first_relevant_rank']} — {question}"
        )
    routed_retrieval = four_rescue["metrics"]["retrieval"]
    lines.extend([
        "",
        f"The four-of-five rule retains the always-MQ2 Hit@3/5 values ({routed_retrieval['hit_at_3']:.4f}/{routed_retrieval['hit_at_5']:.4f}) because its missed monoT5 improvement changes rank 3 to rank 2 without crossing those cutoffs. That coincidence should not be mistaken for a generally safe miss.", "",
        "## 10C-1 conclusion", "",
        "No cheap signal is selective enough to justify a router yet. Capturing all five improvements still escalates 61.1% of questions. Capturing four requires 35.6% escalation and 28 unnecessary LLM calls for four useful ones. This lowers estimated average latency versus always-MQ2, but routing precision remains only 12.5%.", "",
        "Phase 10C-2 is worth revisiting only after out-of-sample validation or collection of the missing dense/BM25 agreement signals. Implementing the current fitted threshold would encode benchmark leakage as production logic.", "",
        "## Limitations and next step", "",
    ])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend([
        "", "Review whether any simple signal captures four or five helpful cases at a materially lower escalation rate. Even a visually strong rule must be checked with leave-one-positive-out stability and a new question set before Phase 10C-2 implements a router.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase10b-results", type=Path, default=DEFAULT_PHASE10B_RESULTS)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    phase10b = json.loads(args.phase10b_results.read_text(encoding="utf-8"))
    report = analyze_phase10b(phase10b, source_path=str(args.phase10b_results))
    write_json(report, args.json_output, args.phase10b_results)
    write_csv(report, args.csv_output)
    write_markdown(report, args.markdown_output)
    print(f"JSON analysis: {args.json_output}")
    print(f"CSV signals: {args.csv_output}")
    print(f"Markdown report: {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
