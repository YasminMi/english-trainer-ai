# Retrieval evaluation

`rag_golden.jsonl` is a hand-labelled question set for the grounded tutor. Every
RAG change is measured against it before it is merged.

```bash
english-trainer eval-rag                      # real embeddings (Ollama, cached)
english-trainer eval-rag --embedder hashing   # deterministic, no Ollama (CI)
english-trainer eval-rag --embedder hashing --retrieval dense
english-trainer eval-rag --embedder hashing --retrieval hybrid
english-trainer eval-rag --json --min-hit-rate 0.8
```

## Dataset

| Kind | Cases | Purpose |
|---|---:|---|
| Answerable | 40 | grammar rules, error corrections, paraphrases, vocabulary, PT and EN |
| Off-topic | 12 | questions the tutor must refuse (includes one prompt-injection attempt) |
| Unsupported | 3 | English questions the notes do not cover (present perfect, passive voice) |

`relevant` lists every chunk label that answers the question. A label is
`<note path>#<section>` for Markdown notes and the JSON anchor otherwise
(`data/content.json#to_be_exercises`). Labels survive re-chunking, unlike chunk ids.

## Metrics

| Metric | Meaning |
|---|---|
| `hit@k` | share of answerable questions with at least one relevant chunk in the top *k* |
| `recall@k` | share of all relevant labels found in the top *k*, averaged |
| `mrr` | mean reciprocal rank of the first relevant chunk |
| `false_accept_rate` | off-topic/unsupported questions that passed the relevance gate |
| `false_reject_rate` | answerable questions the relevance gate refused |

## Results

All results use the 55-case set and `k = 5`. The first two rows use
`embeddinggemma`; the comparison rows use `hashing-512`.

| Change | hit@1 | hit@5 | MRR | False accept | False reject |
|---|---:|---:|---:|---:|---:|
| Baseline: dense search + lexical boost, gate at cosine 0.20 | 0.750 | 0.850 | 0.800 | 0.933 | 0.000 |
| Group `to be` explanations by rule (69 → 57 chunks) | 0.700 | 0.900 | 0.790 | 0.933 | 0.000 |
| Dense + lexical boost, hashing-512 (57 chunks) | 0.300 | 0.475 | 0.375 | 1.000 | 0.000 |
| Hybrid BM25 + dense RRF, hashing-512 (57 chunks) | 0.350 | 0.725 | 0.477 | 1.000 | 0.000 |

Grouping the rule chunks lowers hit@1 only because the 14 near-duplicate
chunks used to crowd the first position with the same label (`tb-01`, `tb-05`
move from rank 1 to 2); two previously missed questions are now found.
Use `--report run.json` on two branches to diff ranks per case.

The baseline gate accepts almost every off-topic question: `embeddinggemma`
scores unrelated text around 0.23–0.35, so a 0.20 threshold never refuses.

The hashing comparison uses the deterministic `hashing-512` embedder and a zero
cosine threshold, so its false accept rate is 1.000 in both modes. Hybrid is now
the default because hit@5 and MRR improve without increasing false accepts in
this comparison. `--retrieval dense` retains the previous ranking. In both
modes, the relevance gate checks each candidate's dense cosine, not its boosted
or fused ranking score. Hybrid searches the same 57 indexed chunks with BM25
and retrieves dense scores for the full corpus before fusing ranks (RRF k=60).
