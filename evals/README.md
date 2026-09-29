# Retrieval evaluation

`rag_golden.jsonl` is a hand-labelled question set for the grounded tutor. Every
RAG change is measured against it before it is merged.

```bash
english-trainer eval-rag                      # real embeddings (Ollama, cached)
english-trainer eval-rag --embedder hashing   # deterministic, no Ollama (CI)
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

Measured with `embeddinggemma` on the 55-case set, `k = 5`.

| Change | hit@1 | hit@5 | MRR | False accept | False reject |
|---|---:|---:|---:|---:|---:|
| Baseline: dense search + lexical boost, gate at cosine 0.20 | 0.750 | 0.850 | 0.800 | 0.933 | 0.000 |

The baseline gate accepts almost every off-topic question: `embeddinggemma`
scores unrelated text around 0.23–0.35, so a 0.20 threshold never refuses.
