#!/usr/bin/env python3
"""Author self-audit of `parse_context_facts`' extraction precision.

`branchmem/benchmark/mab_extension.py` reports 99.9% *coverage* (fraction of
context lines the regex parser matches at all) but never *precision*
(fraction of matched triples that are actually correct extractions of their
source line) -- coverage alone doesn't establish that a matched triple's
entity/predicate/value were pulled out correctly, only that some template
matched.

This script draws a reproducible random sample of parsed triples (seeded,
so re-running reproduces the same sample) and writes them, with their exact
source context line, to `results/final/parser_precision_self_audit.json`
for manual reading. This is explicitly an AUTHOR SELF-AUDIT: the person
running this script reads each source line against the extracted triple and
records whether they match. It is NOT independent human validation (no
second, disinterested annotator was involved) and NOT automated
verification (no oracle exists to check against) -- both are stated
plainly in the output JSON's `method` field and must be described the same
way in the paper. No API calls; only a public dataset download.

Usage: source .venv/bin/activate && PYTHONPATH=. python
scripts/audit_parser_precision.py --n 50 --seed 12345
Then manually inspect the written `sampled_triples` against their
`raw_line` and fill in `n_errors_found` / `precision_point_estimate` /
`precision_95pct_clopper_pearson_ci` by hand.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from branchmem.benchmark.mab_extension import load_mab_conflict_resolution, parse_context_facts

OUT_PATH = Path(__file__).resolve().parents[1] / "results" / "final" / "parser_precision_sample.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    df = load_mab_conflict_resolution()
    rng = random.Random(args.seed)
    all_triples = []
    for ridx, row in df.iterrows():
        parsed, _n_matched, _n_total = parse_context_facts(row["context"])
        lines = [ln for ln in row["context"].strip().split("\n") if ln.strip()]
        for p in parsed:
            raw_line = lines[p.line_index] if p.line_index < len(lines) else None
            all_triples.append({
                "row": int(ridx), "line_index": p.line_index, "entity": p.entity,
                "predicate": p.predicate, "value": p.value, "raw_line": raw_line,
            })

    rng.shuffle(all_triples)
    sample = all_triples[: args.n]
    OUT_PATH.write_text(json.dumps({
        "seed": args.seed, "n_sampled": len(sample), "n_total_parsed_triples_population": len(all_triples),
        "sampled_triples": sample,
    }, indent=2))
    print(f"Wrote {OUT_PATH} ({len(sample)} sampled triples out of {len(all_triples)} total parsed)")
    print("Next: manually compare each 'raw_line' against its (entity, predicate, value) and "
          "record findings in results/final/parser_precision_self_audit.json (see its 'method' field).")


if __name__ == "__main__":
    main()
