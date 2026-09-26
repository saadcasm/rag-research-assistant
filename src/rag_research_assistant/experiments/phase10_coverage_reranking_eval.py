"""Evaluate deterministic set-level evidence selection for Phase 10J."""

import argparse, json, statistics, sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Optional, Sequence

import numpy as np

from ..application import ApplicationSettings, default_retrieval_config, load_retriever
from ..candidate_generation.depth import CandidateDepthRun, evaluate_depth_run
from ..evaluation import evaluate_retrieval_results, load_evaluation_dataset
from ..evidence_selection.candidate_coverage import sha256_file
from ..evidence_selection.selectors import select_evidence, selection_diagnostics
from ..hybrid import reciprocal_rank_fusion
from ..index import load_index
from ..multihop import load_multihop_dataset, verify_evidence_passages
from ..pipeline import read_jsonl
from ..retrievers import HybridRetriever, RerankingRetriever


MULTIHOP = Path("data/evaluation/questions-phase-10g-multihop-draft.jsonl")
REGRESSION = Path("data/evaluation/questions-phase-7.5.jsonl")
INDEX = Path("data/processed/corpus-52/index-legacy")
OUTPUT_JSON = Path("data/evaluation/benchmarks/phase-10j-coverage-reranking-results.json")
OUTPUT_MD = Path("docs/phase-10j-coverage-reranking-results.md")
DEPTHS = (20, 50)
STRATEGIES = {
    "baseline": (0.0, 0.0),
    "diversity": (0.4, 0.0),
    "coverage": (0.0, 0.35),
    "coverage_diversity": (0.2, 0.35),
}


def _pool(question, hybrid, reranker, depth):
    started = perf_counter(); dense = tuple(hybrid.dense.search(question, top_k=depth)); dense_s = perf_counter()-started
    started = perf_counter(); bm25 = tuple(hybrid.lexical.search(question, top_k=depth)); bm25_s = perf_counter()-started
    started = perf_counter(); fused = tuple(reciprocal_rank_fusion([dense, bm25], top_k=depth, rrf_k=60)); rrf_s = perf_counter()-started
    started = perf_counter(); scored = tuple(reranker.rerank(question, fused, depth)); rerank_s = perf_counter()-started
    return dense, bm25, fused, scored, {"candidate_generation": dense_s+bm25_s+rrf_s, "cross_encoder": rerank_s}


def _condition(example, depth, pool, vectors, strategy, weights):
    dense, bm25, fused, scored, latency = pool
    selected = select_evidence(example.question, scored, vectors, top_k=10, strategy=strategy,
                               diversity_weight=weights[0], coverage_weight=weights[1], anchor_top_1=True)
    run = CandidateDepthRun(example.question, depth, dense, bm25, scored, selected.selected,
                            {"dense": 0.0, "bm25": 0.0, "rrf": 0.0, "reranker": latency["cross_encoder"], "total": 0.0},
                            len(scored), 10)
    evaluated = evaluate_depth_run(example, run)
    evaluated.update({
        "selection": selected.to_dict(),
        "selection_diagnostics": selection_diagnostics(selected.selected, vectors),
        "latency_seconds": {**latency, "selector": selected.latency_seconds,
                            "total": latency["candidate_generation"] + latency["cross_encoder"] + selected.latency_seconds},
    })
    return evaluated


def _multi_summary(records, depth, strategy):
    rows = [record["conditions"][str(depth)][strategy] for record in records]
    result = {}
    for k in (3, 5, 10):
        values = [row["final_coverage"][str(k)] for row in rows]
        result[str(k)] = {
            "full": sum(v["full"] for v in values)/len(values),
            "partial": sum(v["partial"] for v in values)/len(values),
            "average_recovered_hops": sum(v["recovered_hops"] for v in values)/len(values),
            "zero": sum(v["recovered_hops"] == 0 for v in values),
            "one": sum(v["recovered_hops"] == 1 for v in values),
            "all": sum(v["full"] for v in values),
        }
    result["oracle_full"] = sum(row["oracle_candidate_coverage"]["full"] for row in rows)/len(rows)
    result["mean_selector_seconds"] = statistics.fmean(row["latency_seconds"]["selector"] for row in rows)
    result["mean_total_seconds"] = statistics.fmean(row["latency_seconds"]["total"] for row in rows)
    return result


def _movement(base, compared):
    before, after = base["final_coverage"]["10"], compared["final_coverage"]["10"]
    outcome = "improved" if after["recovered_hops"] > before["recovered_hops"] else "degraded" if after["recovered_hops"] < before["recovered_hops"] else "unchanged"
    return {"outcome": outcome, "before_hops": before["recovered_hops"], "after_hops": after["recovered_hops"],
            "rescued_hops": sorted(set(after["recovered_hop_indexes"])-set(before["recovered_hop_indexes"])),
            "regressed_hops": sorted(set(before["recovered_hop_indexes"])-set(after["recovered_hop_indexes"]))}


def _regression_metrics(values):
    scored = [v for v in values if v.answerability != "unanswerable"]
    def avg(field):
        rows=[float(getattr(v,field)) for v in scored if getattr(v,field) is not None]; return statistics.fmean(rows) if rows else None
    ranks=[v.first_correct_rank for v in scored if v.first_correct_rank is not None]
    return {"hit_at_1":avg("hit_at_1"),"hit_at_3":avg("hit_at_3"),"hit_at_5":avg("hit_at_5"),
            "recall_at_1":avg("expected_source_recall_at_1"),"recall_at_3":avg("expected_source_recall_at_3"),"recall_at_5":avg("expected_source_recall_at_5"),
            "mean_first_correct_rank":statistics.fmean(ranks) if ranks else None}


def run_experiment(multihop, regression, hybrid, reranker, vectors, progress=True):
    multi_records=[]
    for i, example in enumerate(multihop,1):
        conditions={}
        for depth in DEPTHS:
            pool=_pool(example.question,hybrid,reranker,depth)
            conditions[str(depth)]={name:_condition(example,depth,pool,vectors,name,w) for name,w in STRATEGIES.items()}
        movements={str(depth):{name:_movement(conditions[str(depth)]["baseline"],conditions[str(depth)][name]) for name in STRATEGIES if name!="baseline"} for depth in DEPTHS}
        multi_records.append({"question_id":example.id,"question":example.question,"answerability":example.answerability,"category":example.category,"conditions":conditions,"movements":movements})
        if progress: print(f"[multi {i}/{len(multihop)}] {example.id}",flush=True)
    multi_summary={str(d):{s:_multi_summary(multi_records,d,s) for s in STRATEGIES} for d in DEPTHS}
    movement_summary={str(d):{s:{o:[r["question_id"] for r in multi_records if r["movements"][str(d)][s]["outcome"]==o] for o in ("improved","unchanged","degraded")} for s in STRATEGIES if s!="baseline"} for d in DEPTHS}
    reg_records=[]; reg_evals={(d,s):[] for d in DEPTHS for s in STRATEGIES}
    for i, example in enumerate(regression,1):
        conditions={}
        for depth in DEPTHS:
            pool=_pool(example.question,hybrid,reranker,depth); conditions[str(depth)]={}
            for name,w in STRATEGIES.items():
                selected=select_evidence(example.question,pool[3],vectors,top_k=10,strategy=name,diversity_weight=w[0],coverage_weight=w[1],anchor_top_1=True)
                ev=evaluate_retrieval_results(example,list(selected.selected)); reg_evals[(depth,name)].append(ev)
                conditions[str(depth)][name]={"selected":selected.to_dict(),"first_correct_rank":ev.first_correct_rank}
        reg_records.append({"question_id":example.id,"question":example.question,"answerability":example.answerability,"conditions":conditions})
        if progress: print(f"[regression {i}/{len(regression)}] {example.id}",flush=True)
    reg_summary={str(d):{s:_regression_metrics(reg_evals[(d,s)]) for s in STRATEGIES} for d in DEPTHS}
    return multi_records,multi_summary,movement_summary,reg_records,reg_summary


def write_report(report,path_json,path_md):
    path_json.parent.mkdir(parents=True,exist_ok=True); path_json.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=["# Phase 10J: Coverage-Aware Evidence-Set Selection","","> Experimental only; production is unchanged.","","## Multi-hop coverage","","| Depth | Strategy | Oracle | Full@3 | Full@5 | Full@10 | Partial@10 | Selector ms | Total s |","|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in DEPTHS:
        for s in STRATEGIES:
            row=report["summary"]["multihop"][str(d)][s]
            lines.append(f"| {d} | {s} | {row['oracle_full']:.4f} | {row['3']['full']:.4f} | {row['5']['full']:.4f} | {row['10']['full']:.4f} | {row['10']['partial']:.4f} | {row['mean_selector_seconds']*1000:.3f} | {row['mean_total_seconds']:.3f} |")
    lines.extend(["","## Frozen regression","","```json",json.dumps(report["summary"]["regression"],indent=2),"```","","## Movements","","```json",json.dumps(report["summary"]["movements"],indent=2),"```","","## Interpretation status","","The generated metrics are reviewed in `docs/phase-10j-coverage-reranking.md`. No selector is promoted; production remains unchanged."])
    path_md.parent.mkdir(parents=True,exist_ok=True); path_md.write_text("\n".join(lines)+"\n",encoding="utf-8")


def _parser():
    p=argparse.ArgumentParser(description=__doc__); g=p.add_mutually_exclusive_group(required=True); g.add_argument("--smoke",action="store_true"); g.add_argument("--benchmark",action="store_true"); return p


def main(argv:Optional[Sequence[str]]=None)->int:
    args=_parser().parse_args(argv); loaded=None
    try:
        settings=ApplicationSettings.from_environment(); loaded=load_retriever(default_retrieval_config(settings))
        if not isinstance(loaded.retriever,RerankingRetriever) or not isinstance(loaded.retriever.base,HybridRetriever): raise ValueError("frozen hybrid reranker required")
        index=load_index(settings.chunks_path,INDEX)
        if index.model_name!=loaded.metadata.embedding_model: raise ValueError("embedding index model mismatch")
        vectors={chunk.chunk_id:np.asarray(index.embeddings[i],dtype=np.float32) for i,chunk in enumerate(index.chunks)}
        multi=load_multihop_dataset(MULTIHOP); issues=verify_evidence_passages(multi,index.chunks)
        if issues: raise ValueError(f"evidence validation failed: {issues[:3]}")
        if args.smoke:
            records,summary,movements,_,_=run_experiment(multi[:1],[],loaded.retriever.base,loaded.retriever.reranker,vectors)
            print(json.dumps({"question":records[0],"summary":summary,"movements":movements},indent=2)); return 0
        regression=load_evaluation_dataset(REGRESSION)
        records,summary,movements,reg_records,reg_summary=run_experiment(multi,regression,loaded.retriever.base,loaded.retriever.reranker,vectors)
        report={"schema_version":1,"experiment":"phase-10j-coverage-reranking","created_at":datetime.now(timezone.utc).isoformat(),
                "settings":{"depths":list(DEPTHS),"strategies":{k:{"diversity_weight":v[0],"coverage_weight":v[1]} for k,v in STRATEGIES.items()},"top_k":10,"top_1_anchored":True,"production_changed":False,"gold_used_for":"evaluation_only"},
                "provenance":{"multihop_sha256":sha256_file(MULTIHOP),"regression_sha256":sha256_file(REGRESSION),"chunks_sha256":sha256_file(settings.chunks_path)},
                "summary":{"multihop":summary,"movements":movements,"regression":reg_summary},"multihop_questions":records,"regression_questions":reg_records}
        write_report(report,OUTPUT_JSON,OUTPUT_MD); print(f"JSON report: {OUTPUT_JSON}\nMarkdown report: {OUTPUT_MD}"); return 0
    except (OSError,ValueError,KeyError,json.JSONDecodeError) as exc: print(f"Error: Phase 10J failed: {exc}",file=sys.stderr); return 2
    finally:
        if loaded is not None: loaded.close()


if __name__=="__main__": raise SystemExit(main())
