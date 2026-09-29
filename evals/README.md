# Retrieval evaluation

The original 55 cases in `rag_golden.jsonl` were used to tune the gate. Ten
later cases have the `held-out` tag and were not used to choose the threshold.

```bash
english-trainer eval-rag --retrieval dense --subset core --json
english-trainer eval-rag --retrieval hybrid --subset held-out --json
english-trainer eval-rag --retrieval dense --dump-scores scores.jsonl
english-trainer eval-rag --embedder hashing --json
```

`--subset` accepts `all` (default), `core`, or `held-out`. `--dump-scores`
writes one JSON object per case with its label kind, top dense cosine,
top-1 minus top-2 dense margin, highest corpus BM25 score, query-term
overlap in the best BM25 chunk, and the terms. `--report run.json` writes
ranks and gate decisions. The hashing embedder is the deterministic CI path.

## Dataset and metrics

| Partition | Answerable | Off-topic | Unsupported |
|---|---:|---:|---:|
| Original tuning set | 40 | 12 | 3 |
| Held-out additions | 4 | 4 | 2 |

The `relevant` field lists every chunk label that answers the question. A
label is a Markdown note path and section or a JSON anchor. An empty list
means the tutor should refuse. `hit@k` and `recall@k` measure answerable
retrieval, MRR measures the first relevant rank, false accept measures
unanswerable questions that pass the gate, and false reject measures
answerable questions refused by the gate.

## Score diagnosis and threshold choice

Original 55 cases, embeddinggemma, 57 chunks. Each range shows minimum / median / maximum. BM25 and overlap use the best BM25 chunk; overlap removes common Portuguese and English function words.

| Kind | Top cosine | Top-2 margin | BM25 top | Term overlap |
|---|---|---|---|---|
| Answerable (40) | .324 / .527 / .744 | .000 / .040 / .176 | 4.459 / 9.488 / 18.337 | .000 / .500 / .833 |
| Off-topic (12) | .199 / .282 / .376 | .001 / .013 / .065 | 3.835 / 5.202 / 7.456 | .000 / .200 / .500 |
| Unsupported (3) | .406 / .452 / .503 | .001 / .017 / .058 | 5.685 / 10.780 / 17.140 | .167 / .571 / .600 |

Threshold sweep on the original set, using the top dense cosine:

| Minimum cosine | False accept | False reject |
|---:|---:|---:|
| .20 (old default) | .933 | .000 |
| .35 | .267 | .050 |
| **.38 (new default)** | **.200** | **.050** |
| .40 | .200 | .075 |
| .45 | .133 | .150 |

The gate accepts a retrieved candidate only when its raw dense cosine is at least .38. It ignores lexical boosts and RRF ranking scores. The threshold is configurable with RAG_MIN_SCORE; the hashing CI eval still uses zero. A margin or lexical cutoff would add complexity without solving the unsupported grammar overlap: those three questions score .406–.503 and two have strong BM25 and term overlap. Their margins also overlap answerable questions. The threshold was chosen on the original set and left unchanged after viewing the held-out set.

## Results

Real embeddinggemma via local Ollama, k = 5. The old and new rows differ only in the cosine threshold (.20 versus .38). Retrieval ranking metrics do not change with the gate.

| Partition | Mode | Gate | hit@1 | hit@5 | MRR | False accept | False reject |
|---|---|---|---:|---:|---:|---:|---:|
| Original 55 | Dense | .20 | .700 | .900 | .790 | .933 | .000 |
| Original 55 | Dense | **.38** | .700 | .900 | .790 | **.200** | **.050** |
| Original 55 | Hybrid | .20 | .700 | .900 | .780 | .800 | .000 |
| Original 55 | Hybrid | **.38** | .700 | .900 | .780 | **.200** | **.050** |
| Held-out 10 | Dense | .20 | .250 | .750 | .500 | 1.000 | .000 |
| Held-out 10 | Dense | **.38** | .250 | .750 | .500 | **.333** | **.000** |
| Held-out 10 | Hybrid | .20 | .750 | 1.000 | .875 | 1.000 | .000 |
| Held-out 10 | Hybrid | **.38** | .750 | 1.000 | .875 | **.333** | **.250** |
| Full 65 | Dense | .20 | .659 | .886 | .763 | .952 | .000 |
| Full 65 | Dense | **.38** | .659 | .886 | .763 | **.238** | **.045** |
| Full 65 | Hybrid | .20 | .705 | .909 | .789 | .857 | .000 |
| Full 65 | Hybrid | **.38** | .705 | .909 | .789 | **.238** | **.068** |

The held-out false accepts are both unsupported grammar questions; all four held-out off-topic questions are refused. The held-out hybrid false reject has a relevant chunk at rank 1, but its cosine is below .38. Hybrid improves full-set retrieval ranks but has a higher false reject rate, so dense remains the default. A cosine gate cannot verify that a grammatically related chunk actually contains the requested rule; those unsupported cases need stronger evidence than the current retrieval signals provide.
