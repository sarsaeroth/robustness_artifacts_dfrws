# Certified Robustness and Code Localization of Graph-Based Malware Classifiers

Artifact for the paper. It covers both the 10-page version and the extended version: the result file
behind every number, the scripts that turn those files into the papers' numbers, tables and figures, one
command per quantitative claim of both versions that recomputes it from those files, and the application
lists needed to rebuild the four corpora.

## Check the paper

    ./install.sh    # checks for Python 3.9+, installs matplotlib into ./.venv
    ./verify.sh

`verify.sh` first regenerates the outputs: `scripts/robustness_numbers.py` reads the result files in
`data/results/` and writes `regenerated/numbers.json`, the tables of the extended version
(`regenerated/tables_extended/*.tex`) and the figures (`regenerated/figures/`), and
`scripts/robustness_10p_tables.py` writes the tables of the 10-page version (`regenerated/tables_10p/*.tex`).
Then it runs a flat list of commands, one per number of the papers (in the order of the 10-page version, then
the extended version, section by section), and one `cat` per table where the paper includes it. Every command is
printed as `$ <command line>` followed by its output, and the whole output is saved to `verify_output.txt`:

    $ .venv/bin/python scripts/query.py value regenerated/numbers.json::proxy.n --sep
    <the number>
    $ .venv/bin/python scripts/query.py stat max regenerated/numbers.json::proxy.ident.median_signed_error_max ... --digits 3
    <the number>

The output contains no labels: the command says what is computed and from which file. To check the paper, find
a number in the paper, find the command that reads the corresponding value (the comments in `verify.sh` mark the
paper version and section of each group of commands), and compare. Tables are printed as the LaTeX the papers
include, so they can be compared cell by cell. The artifact contains no expected values and compares nothing.
Every command runs even if an earlier one fails; `verify.sh` exits non-zero if any command failed (a missing or
malformed input file), never because of a value. It uses `.venv` if `./install.sh` created it, else `$PYTHON`,
else `python3`.
A run takes under a minute. Any command can be run on its own from this directory after `verify.sh` has
regenerated `regenerated/`.

### Reading a command

Every command is `scripts/query.py`, the generic tool shared by the three artifacts of this repository (the same
file, byte for byte), or a `cat` of a regenerated table. This artifact has no `derive.py`: its paper-specific
computations are `robustness_numbers.py` and `robustness_10p_tables.py`, which assemble `numbers.json` and the
tables from the result files. `python3 scripts/query.py --help` (and `query.py OPERATION --help`) documents every
operation and option.

- The subcommand is the operation: `value` (one value), `stat min|max|median|mean|sum|count|max-abs` (over every
  value the SPECs name), `diff` / `ratio` (of two values), `diff-stat` / `ratio-stat STAT A B A B ...` (a
  statistic over pairwise differences or ratios), `above-one STAT` (over each value minus 1), `nth-largest N`,
  `largest-share N` (the N largest values over the total), `weighted-mean VALUES WEIGHTS`; on records `count`
  (rows or records passing `--where`, `--where-in`, `--below`, `--above`; distinct values with `--distinct`),
  `field-stat STAT RECORDS --field F`, `group-size RECORDS --by F` (with `--over-groups STAT`,
  `--exclude-largest`, `--largest-share N`) and `group-stat STAT RECORDS --by F --field F --over-groups STAT`.
- A SPEC names values: a file path, then the keys or list indices to follow inside it, joined by `::`.
  `regenerated/numbers.json::proxy.tightness::labeled::k_grad_median` is the key `k_grad_median` of `labeled` of
  the entry `proxy.tightness` of `numbers.json`. A key `*` expands to every key of a dict or every item of a list,
  so `regenerated/numbers.json::proxy.mx::*::M5_spearman` names that field for every explainer; glob keys
  (`flip_*`, `labeled|*`) expand to the matching keys; a CSV file is the list of its rows; null values are
  skipped; `--skip-key KEY` leaves out the entries named KEY when a pattern expands a dict. SPECs with `*`, `|`
  or spaces are quoted in the printed command.
- The format flags only set the precision of the printed number: `--digits N` (rounded half up; negative N
  rounds to tens or hundreds; `--rounding half-even` where the paper's value repeats a table cell, whose medians
  are rounded half to even, and `--rounding ceiling` where the paper states an upper bound), `--percent` (times
  100, without the `%` sign), `--permille` (times 1000), `--sep` (thousands separators), `--sci N` /
  `--significant N` (scientific notation, significant digits). Signs are not printed for positive values.
- A range in the paper ("4% to 9%") is two commands, `min` and `max`; a list of values is one command each.

Every printed value is a result recorded in the shipped files or computed from them: results from
`data/results/` (mostly through `numbers.json`, which `robustness_numbers.py` assembles from them), corpus
sizes, class counts and grouping counts from `data/manifests/`, and the protocol parameters that a result file
records (hash dimension, surrogate attack budget, numbers of donors). Numbers the papers take from other
sources, or that no shipped file records, have no command, because the artifact cannot compute them:

- citations, standard and rule numbers;
- facts about the source datasets reported by the papers that published them (the size of earlier collections,
  how many applications of them remain obtainable or were kept, the number of released pairs, marker rates,
  results quoted from the corpus paper, and similar counts stated with a citation);
- sampling targets that differ from the analyzable counts in the manifests (the manifests list the
  applications actually analyzed, and the commands count those), and counts the papers obtain by adding or
  subtracting counts of different sources (the staged subset of providers plus benign donors, the Popa builds
  without an extractable graph);
- the thresholds of the MalNet payload-namespace selection and the full graph sizes of the MalNet payload
  families, whose script and graphs are not released, and approximate sizes used only for illustration;
- the protocol settings listed below.

### Settings (not printed)

Protocol settings that no result file records as a value are set in the experiment scripts
(`scripts/experiments/`) and are not printed. Some also appear inside result key names or descriptions, noted in
the last column.

| setting | where it is set | also visible in |
|---|---|---|
| cosine threshold of the lineage (near-duplicate) grouping | `NEARDUP_COS` in `exp_harness.py` | |
| shares of top-ranked methods deleted in the faithfulness tests | `KFRAC` in `exp_p3_explainer_mx.py`, `exp_p4_cfg_deletion.py` | keys such as `M3_flip_0.01` |
| size of the top set compared for bootstrap stability | `exp_p3_explainer_mx.py` | key `M4_top100_jaccard` |
| stub limit of the graph-level attack | `MAX_K` in `exp_p3_steer_graph.py` | |
| random padding budget of the frontier | `MAX_T` in `exp_p3_frontier.py` | |
| bundling budget | `BUDGET` in `exp_bundle_attack.py` | |
| block-repetition stub settings | `SETTINGS` in `exp_p4_cfg_steer.py` | keys such as `concat|b16` |
| per-graph and total vector dimension of the deployed CFG+FCG classifier | `D` in `exp_p4_cfg_cert.py` | |
| forest size and MLP hidden units of the model ladder | `RandomForestClassifier`, `MLPClassifier` arguments in `exp_l1_ladder.py` | |
| hidden units and training epochs of the identifier GIN | `HID`, `EPOCHS` in `exp_tier3_gnn.py` | |
| size of the certified random forest | `exp_p3_tree_cert.py` | the `model` text of `tree_cert.json` |
| edit sites per application and colorings of the perturbation-bound check | `NOPS`, `COLOURS` in `exp_p3_thm_bounds.py` | |
| operations, locales, budgets and seeds of the bounded perturbation | `OPS`, `LOCALES`, `BUDGETS`, `SEEDS` in `exp_r4_perturb.py` | |
| upper percentile of the occupancy ratio and isolation level of the dimension estimate | `exp_tier1_identifiability.py` | keys `rho_p95`, `d_star_90_median` |
| Jaccard thresholds of the host-cluster holdout | `exp_sdk_transplant.py` | keys of `A2b` |
| share of known applications the Play threshold keeps | `exp_wild_colorings.py`, `exp_open_world.py` | key `wild_fp_at_tau95` |

To regenerate only the outputs:

    python3 scripts/robustness_numbers.py          # numbers.json, extended tables, figures
    python3 scripts/robustness_10p_tables.py       # tables of the 10-page version

`ROBUSTNESS_DATA` and `ROBUSTNESS_OUT` override their input and output directories (the commands of `verify.sh`
read `data/` and `regenerated/`).

## Contents

    data/results/datasets/<corpus>/   result files per corpus and setting, as written by the experiments:
                                        proxy_v3_gneardup_fcg   proxy corpus, lineage grouping, FCG (the common protocol)
                                        proxy_v3_gneardup       proxy corpus, lineage grouping, CFG and FCG
                                        proxy_v3                proxy corpus, DEX grouping (control)
                                        malnet, piggyback, cic_andmal          the comparison corpora
                                        *_payload               analyses restricted to applications with payload ground truth
                                        *_cfg+fcg, *_payload_cfg+fcg           CFG-only and deployed-model analyses
    data/results/cross_dataset/       pooled separability (RQ4) and the Google Play results per coloring
    data/results/extra/               derived files for the corpus description: CFG sizes, the proxy corpus's min-cut
                                      results and per-provider FCG sizes, and its DEX file counts
    data/manifests/                   one CSV per corpus (see below) and the Google Play sample
    scripts/robustness_numbers.py     result files -> numbers.json, tables of the extended version, figures
    scripts/robustness_10p_tables.py  numbers.json -> tables of the 10-page version
    scripts/experiments/              the experiment scripts that produced the result files, and their modules
    scripts/query.py                  the generic tool shared by the three artifacts: values, statistics, pairs,
                                      record counts, group statistics and tables out of the data files (see --help)
    verify.sh                         regenerates the outputs, then one command per number of both versions

The experiment scripts are shipped to document the protocol; they are rewritten in a flat style for readability, with the logic as run. They expect the full
research tree (application graphs, vectors and staging directories that are not released) and do not
run from this artifact.

## Corpora

Each manifest lists the applications of one corpus by SHA-256 with its label:

| file | corpus | columns |
|---|---|---|
| `proxy.csv` | residential-proxy corpus (one label per provider) and the applications without a known proxy SDK (label `non_proxy`) | `closed_world` (1 for the applications of the classification task), `dex_group`, `lineage_group_closed`, `lineage_group_open` |
| `malnet.csv` | the MalNet families used and the benign applications | as above, without `dex_group` |
| `piggyback.csv` | original and piggybacked applications of Li et al.'s pairs | as above |
| `cic_andmal2020.csv` | CIC-AndMal2020 adware, riskware, potentially unwanted and trojan applications, and benign applications | as above |
| `wild_play.csv` | the Google Play applications sampled uniformly from AndroZoo | `proxy_marker_found` (1: carries a proxy SDK marker, excluded), `fcg_vector` (1: has an FCG vector) |

The commands of `verify.sh` compute every corpus size, class count and group count from these files.
`lineage_group_closed` is the grouping used for cross-validation of the classification task (no benign
class) and `lineage_group_open` the grouping with the benign class: applications that share a package
name or whose identifier-coloring FCG vectors are near-duplicates (cosine similarity at or above the
threshold the paper states), joined transitively.

## Not included, and how to re-obtain it

- **Applications.** APKs are third-party software and are not redistributed. All applications can be
  retrieved by SHA-256 from AndroZoo, which requires an API key from the AndroZoo project; MalNet
  family labels come from MalNet, the pairs from Li et al.'s piggybacking dataset and the categories
  from CCCS-CIC-AndMal2020.
- **Graphs, WL vectors and token caches.** They are derived from the applications, are tens of
  gigabytes, and are not needed to check the papers. The proxy corpus's identifier and combined
  vectors are released with the corpus paper's artifact.
- **Per-method ground truth.** Proxy SDK namespaces come from the corpus and reverse-engineering
  papers; MalNet payload namespaces and piggyback riders are recomputed by
  `scripts/experiments/piggyback_payload.py` and the localization scripts from the applications.
