"""``english-trainer eval-rag``: measure retrieval quality on the gold question set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from english_trainer.config import Settings
from english_trainer.rag.chunking import build_knowledge_chunks
from english_trainer.rag.embeddings import (
    CachingEmbedder,
    Embedder,
    HashingEmbedder,
    OllamaEmbeddingClient,
)
from english_trainer.rag.evaluation import EvalReport, evaluate, load_cases
from english_trainer.rag.index import KnowledgeIndex
from english_trainer.rag.retrieval import DenseRetriever, HybridRetriever
from english_trainer.rag.vector_store import QdrantKnowledgeStore
from english_trainer.storage.content import ContentRepository

DEFAULT_DATASET = Path("evals") / "rag_golden.jsonl"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, help=f"JSONL gold set (default: {DEFAULT_DATASET})")
    parser.add_argument("-k", type=int, default=5, help="cut-off rank for hit@k and recall@k")
    parser.add_argument(
        "--retrieval",
        choices=["dense", "hybrid"],
        help="retrieval mode (default: RAG_RETRIEVAL, or hybrid)",
    )
    parser.add_argument(
        "--embedder",
        choices=["ollama", "hashing"],
        default="ollama",
        help="'hashing' is deterministic and needs no Ollama (used in CI)",
    )
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="re-embed everything (default: reuse vectors cached under data/.cache)",
    )
    parser.add_argument(
        "--report", type=Path, help="write per-case results as JSON (useful to diff two runs)"
    )
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=None,
        help="exit with status 1 when hit@k falls below this value (quality gate)",
    )


def run(args: argparse.Namespace, settings: Settings) -> int:
    paths = settings.paths
    dataset = args.dataset or paths.home / DEFAULT_DATASET
    cases = load_cases(dataset)

    embedder: Embedder
    if args.embedder == "hashing":
        embedder = HashingEmbedder()
    else:
        embedder = OllamaEmbeddingClient(
            model=settings.rag.embed_model, base_url=settings.ollama.base_url
        )
        if not args.no_cache:
            embedder = CachingEmbedder(embedder, paths.data_dir / ".cache" / "embeddings")
    content = ContentRepository(paths).load()
    chunks = build_knowledge_chunks(content, paths.notes_dir, paths.home)

    store = QdrantKnowledgeStore(None)
    try:
        KnowledgeIndex(embedder, store).sync(chunks, force=True)
        min_score = settings.rag.min_score if args.embedder == "ollama" else 0.0
        mode = args.retrieval or settings.rag.retrieval
        retriever = (
            HybridRetriever(embedder, store)
            if mode == "hybrid"
            else DenseRetriever(embedder, store, oversample=1)
        )
        if isinstance(retriever, HybridRetriever):
            retriever.set_chunks(chunks)
        report = evaluate(
            retriever,
            cases,
            k=args.k,
            gate=lambda hits: any(
                hit.dense_score is not None and hit.dense_score >= min_score for hit in hits
            ),
        )
    finally:
        store.close()

    _print(report, as_json=args.json, chunks=len(chunks), embedder=embedder.model)
    if args.report:
        args.report.write_text(
            json.dumps({"summary": report.summary(), "cases": report.cases()}, indent=2),
            encoding="utf-8",
        )
    if args.min_hit_rate is not None and report.hit_rate() < args.min_hit_rate:
        print(
            f"Quality gate failed: hit@{args.k} {report.hit_rate():.3f} < {args.min_hit_rate}",
            file=sys.stderr,
        )
        return 1
    return 0


def _print(report: EvalReport, *, as_json: bool, chunks: int, embedder: str) -> None:
    if as_json:
        print(json.dumps({"embedder": embedder, "chunks": chunks, **report.summary()}, indent=2))
    else:
        print(f"Embedder: {embedder}   Indexed chunks: {chunks}\n")
        print(report.format())
