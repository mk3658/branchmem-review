# REPRODUCIBILITY.md

Exact settings and commands needed to reproduce every number in the paper,
without spending on new API calls (the cache in `llm_cache/` is committed).

## Model and decoding parameters

| Component | Model | Temperature | Max tokens | top\_p |
|---|---|---|---|---|
| `ThreeWayLLMMerge`, `TwoWayLLMMerge`, `RawTextLLMMerge` (locked run) | `gpt-5.4-nano` | `0.0` | `8000`* | provider default (not set) |
| `LLMJudgeConflictDetector` | `gpt-5.4-nano` | `0.0` | `1024` | provider default (not set) |
| Post-hoc `gpt-5.4-mini` robustness check | `gpt-5.4-mini` | `0.0` | `1024` | provider default (not set) |
| Post-hoc cross-vendor checks (`run_cross_vendor_*.py`) | `claude-haiku-4-5-20251001` | `0.0` | `1024`/`8000` | n/a (Anthropic backend) |

\* The original locked run used `max_tokens=1024`/`2048`; raised to `8000`
for the expanded MemoryAgentBench and later ablation runs after discovering
that large batched JSON responses were silently truncated at the lower
limits. All numbers reported in the paper use the corrected `8000` limit for
any call that batches more than ~15 conflicts.

Backend: `branchmem/llm/openai_compatible_backend.py`, OpenAI's chat
completions API. `top_p` is never set explicitly (uses the API's own
default). Newer reasoning-family models sometimes reject `max_tokens` (want
`max_completion_tokens`) or a non-default `temperature`; the backend
detects this via a `BadRequestError` retry (up to 3 attempts) and remembers
the fix per backend instance — this is a client-side compatibility shim, not
a change to the requested decoding parameters themselves.

**API run dates**: locked confirmatory run and pilot, 2026 (see git history
for exact dates); review-response rounds (new baselines, expanded MAB,
`RawTextLLMMerge`, resolvable-category power expansion, abstention
metrics, semantic-resolvable category) run 2026-07-04, per this file's own
edit date.

## Retry policy and failure handling

- Network/API errors: handled by the OpenAI SDK's own retry behavior; the
  backend's 3-attempt loop (above) is specifically for the
  `max_tokens`-vs-`max_completion_tokens` and temperature-support
  incompatibilities, not general transient failures.
- **JSON parsing policy**: `ThreeWayLLMMerge`/`TwoWayLLMMerge`/
  `RawTextLLMMerge` expect a JSON object with a `resolutions` array. If
  `json.loads` fails on the raw response text, every conflict in that batch
  is marked `FLAGGED_UNRESOLVED` rather than guessed — this is a
  conservative fallback, not a silent skip. Historical JSON failure rate:
  zero in the current cache (verified by scanning `llm_cache/*.json` for
  entries whose `text` field fails `json.loads()`).

## Cache key construction

`branchmem/llm/cache.py`: the cache key is
`sha256(json.dumps({backend, model, system, prompt, temperature, max_tokens, ...}, sort_keys=True))`.
Any change to the prompt text, system prompt, model, or decoding parameter
produces a new key — there is no cross-configuration cache reuse. One JSON
file per unique key under `llm_cache/`, containing the full request
parameters, response text, model, and token counts. `find_collisions()` in
`branchmem/merge/base.py` iterates conflict keys in **sorted**, not raw
`set`, order specifically so the JSON built from it — and thus the cache
key — is stable across separate Python process invocations (Python
randomizes `set`/string-hash iteration order per process by default).

## Reproduce without spending anything

```bash
source .venv/bin/activate
python -m pytest tests/ -q                      # unit tests, no API key needed
python scripts/run_experiment.py                # locked Phase 6 run — reads from llm_cache/, $0 if cache populated
```

Every real LLM call across every round of this project (pilot, locked run,
post-hoc checks, new baselines, semantic-resolvable category, balanced
detector benchmark) is committed under `llm_cache/`, so none of the above
commands make a network call as long as the exact prompt/config combination
was already run once.

## Total call volume

The paper's Experimental Setup section reports the headline total for the
locked run (460 calls; 111,403 input / 45,462 output tokens);
later post-hoc rounds add further cached calls under `llm_cache/`, all
released alongside this repository.
