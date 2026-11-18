# Certified Robustness and Code Localization of Graph-Based Malware Classifiers

Artifact for the paper (10-page and extended versions): the result file behind every number, the scripts that
turn them into the papers' numbers and tables, and the application lists of the four corpora.

## Run

    ./install.sh    # checks for Python 3.9+, installs dependencies into ./.venv
    ./verify.sh     # regenerates numbers and tables, then prints every number of both versions

## Output

Pairs of a command (`$ ...`) and its result, in each paper version's order, also written to
`verify_output.txt`. Nothing stores the paper's values: compare each result with the paper.

- `scripts/robustness_numbers.py` and `scripts/robustness_10p_tables.py` regenerate `regenerated/numbers.json`
  and the tables (LaTeX, printed with `cat`).
- `scripts/query.py` (shared by all three artifacts) reads values: a SPEC is a file path followed by keys,
  joined by `::` (`*` expands to every key). `.csv` and `.jsonl` files are lists of rows.

`--help` on each program documents every option.

## Layout

    data/manifests/   one CSV per corpus (proxy, MalNet, piggyback, CIC-AndMal2020): SHA-256, label,
                      closed-world flag, lineage groups; wild_play.csv (Play sample)
    data/results/     result files as the numbers script reads them
    scripts/          the scripts above; experiments/ holds the experiment scripts

## Not included

APKs, graphs and feature vectors (retrieve the applications by SHA-256 from AndroZoo and re-extract them with
the experiment scripts), cited figures, and experiment settings, which are set in `scripts/experiments/`.
