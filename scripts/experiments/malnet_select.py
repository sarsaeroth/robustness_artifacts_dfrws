#!/usr/bin/env python3
"""Pick the MalNet families to pull, balancing breadth against per-family depth.

MalNet labels 1.26M AndroZoo hashes with 696 families. E7 and E8 want breadth
across families; E9 and E10 want depth inside each, since recurrence and variant
structure are not estimable from a handful of samples. The default trades those
at 40 families by 150 samples.

Selection inside a family is spread over the family's hash order rather than
taken from the head, so the sample is not accidentally one contiguous build run.

Env: N_FAMILIES (40), PER_FAMILY (150), BENIGN_N (500)
"""
import collections
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "datasets", "malnet")
N_FAMILIES = int(os.environ.get("N_FAMILIES", "40"))
PER_FAMILY = int(os.environ.get("PER_FAMILY", "150"))
BENIGN_N = int(os.environ.get("BENIGN_N", "500"))

# group label rows by family
by = collections.defaultdict(list)
types = {}
for r in csv.DictReader(open(os.path.join(M, "malnet_labels.csv"))):
    by[r["family"]].append(r["sha256"])
    types[r["family"]] = r["type"]

# rank families by size, keep the largest N_FAMILIES
benign = by.pop("benign", [])
ranked = sorted(by, key=lambda f: -len(by[f]))[:N_FAMILIES]

# spread the selection over each family's hash order
rows = []
for f in ranked:
    hs = sorted(by[f])
    step = max(1, len(hs) // PER_FAMILY)
    for h in hs[::step][:PER_FAMILY]:
        rows.append({"sha256": h, "family": f, "cohort": types[f]})
hb = sorted(benign)
step = max(1, len(hb) // BENIGN_N)
for h in hb[::step][:BENIGN_N]:
    rows.append({"sha256": h, "family": "benign", "cohort": "benign"})

# write the download list
with open(os.path.join(M, "download_list.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["sha256", "family", "cohort"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

# summary
print(f"[*] {len(ranked)} families x up to {PER_FAMILY}, plus {min(len(hb),BENIGN_N)} benign")
print(f"  {'family':20}{'type':16}{'available':>11}{'selected':>10}")
sel = collections.Counter()
for r in rows:
    sel[r["family"]] += 1
for f in ranked[:20]:
    print(f"  {f:20}{types[f]:16}{len(by[f]):>11}{sel[f]:>10}")
if len(ranked) > 20:
    print(f"  ... and {len(ranked)-20} more families")
print(f"[*] total {len(rows)} samples -> {M}/download_list.csv")
