# FitSignal — Intelligent Candidate Discovery & Ranking

Ranks the top-100 candidates from a 100,000-candidate pool against the
**Senior AI Engineer — Founding Team** job description, the way a great recruiter
would: by reading the *role and career history*, not by counting keywords.

## TL;DR approach

The dataset is adversarial by design. We verified empirically that:

- **Skills are noise.** Every skill appears ~12,000 times, uniformly sprayed
  across all candidates. Having "RAG" or "Pinecone" in your skills list means
  nothing. We deliberately **do not** use the skills array as evidence of fit.
- **The signal lives in title + career-history free-text + trajectory +
  behavioral availability signals.**

So we use a **hybrid ranker**:

1. **Semantic match** — a CPU sentence-transformer (`all-MiniLM-L6-v2`) embeds each
   candidate's *summary + headline + career-history descriptions* (not skills) and
   scores them against the JD broken into positive *and negative* facet queries
   (retrieval/ranking/eval vs. research-only/CV-speech/consulting anti-patterns).
2. **Interpretable feature scoring** — a role/title classifier (the most decisive
   signal, with explicit sub-traps for AI-Research, Computer-Vision and Junior-ML
   titles), career-text evidence (production retrieval/ranking/eval), experience
   band, consulting-only and title-chaser penalties, and location fit.
3. **Behavioral multiplier** — recruiter response rate, last-active recency,
   open-to-work, recruiter saves. A perfect-on-paper candidate who has been idle
   6 months with a 5% response rate is down-weighted: not actually hireable.
4. **Honeypot guard** — strict impossibility checks (job duration exceeding time
   since it started, many "expert" skills with 0 months used, end-before-start,
   etc.) force the ~80 planted honeypots to the bottom, keeping us well under the
   10% honeypot disqualification threshold.
5. **Faithful reasoning** — each row's justification is generated purely from facts
   already in that candidate's profile (no hallucination), citing the decisive
   factors and one honest concern, varied across ranks.

No LLM is called during ranking (the contest forbids it). The expensive embedding
is a one-time **offline precompute**; the timed ranking step only loads `.npy`
arrays and does matrix math.

## Setup

```bash
uv sync
```

## Reproduce the submission

```bash
# 1. Offline precompute (one-time; may exceed the 5-min ranking budget)
python precompute.py --candidates ./candidates.jsonl --artifacts ./artifacts

# 2. Ranking step (runs within <=5 min, <=16GB, CPU, no network)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

`rank.py` loads the precomputed embeddings from `./artifacts`. If they are absent
or do not cover the given candidates (e.g. the small sandbox sample), it embeds
inline — fine for <=100 candidates.

## Files

| File | Purpose |
|------|---------|
| `common.py` | Parsing, role classification, feature extraction, honeypot detection, scoring, reasoning |
| `embed.py` | CPU sentence-transformer helper |
| `precompute.py` | Offline: builds candidate + JD embeddings into `./artifacts` |
| `rank.py` | Timed ranking step → `submission.csv` |
| `tools/explore.py` | Streaming data profiler (how we found the traps) |
| `tools/validate_submission.py` | Organizer-provided format validator |
| `deck/` | Pitch deck source (`slides.md`) + PDF generator (`make_deck.py`) |

## Validate

```bash
python tools/validate_submission.py submission.csv
```

## Pitch deck

```bash
uv run --with reportlab python deck/make_deck.py   # -> deck/fitsignal_deck.pdf
```

## Repository layout

```
.
├── rank.py            # timed ranking step -> submission.csv
├── precompute.py      # offline embedding precompute -> artifacts/
├── common.py          # role classifier, features, honeypot guard, scoring, reasoning
├── embed.py           # CPU sentence-transformer helper
├── tools/             # data profiler + format validator
├── deck/              # pitch deck (markdown source + PDF generator)
├── pyproject.toml     # dependencies (managed by uv)
└── submission_metadata.yaml
```
