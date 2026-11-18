#!/usr/bin/env bash
# Regenerate numbers.json, the tables of both paper versions and the figures into regenerated/ from the
# released result files, then recompute the numbers of both versions of the paper.
# Each command is printed as "$ <command line>", followed by its result: one number, or a table. The program,
# the subcommand and the arguments say what is computed from which shipped file. Nothing is compared against
# expected values; read the results next to the paper. The output is also written to verify_output.txt.
# Every command runs even if an earlier one fails; the exit status is non-zero if any command failed.
# Usage: ./verify.sh
set -euo pipefail
cd "$(dirname "$0")"

# interpreter: the artifact's .venv if ./install.sh created it, else $PYTHON, else python3
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY="${PYTHON:-python3}"
fi
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
failed=0

# print the command line (arguments quoted for the shell where needed), then run it; a failure is
# remembered and the list continues
run() {
  local line="$"
  local arg
  local quoted
  for arg in "$@"; do
    case "$arg" in
      "" | *[!A-Za-z0-9_./:=,@+-]*)
        quoted=${arg//"'"/"'\\''"}
        line="$line '$quoted'"
        ;;
      *) line="$line $arg" ;;
    esac
  done
  echo "$line"
  if ! "$@"; then
    failed=1
  fi
}

{
# regenerate numbers.json, the tables of both versions and the figures into regenerated/
run "$PY" scripts/robustness_numbers.py
run "$PY" scripts/robustness_10p_tables.py

# 10-page version: Abstract
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.n --sep
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.n --sep
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.classes
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.n --sep
run "$PY" scripts/query.py value regenerated/numbers.json::cic.n --sep
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.loc1024.ratio_by_explainer::*' 'regenerated/numbers.json::malnet.loc1024.ratio_by_explainer::*' 'regenerated/numbers.json::piggyback.loc1024.ratio_by_explainer::*' --digits 2
run "$PY" scripts/query.py above-one min regenerated/numbers.json::proxy.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::malnet.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::piggyback.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::cic.tightness::labeled::ceiling_over_floor_median --percent --digits 0
run "$PY" scripts/query.py above-one max regenerated/numbers.json::proxy.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::malnet.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::piggyback.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::cic.tightness::labeled::ceiling_over_floor_median --percent --digits 0
run "$PY" scripts/query.py ratio-stat min regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::malnet.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::piggyback.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::cic.tightness::labeled::k_grad_median --digits 0
run "$PY" scripts/query.py ratio-stat max regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::malnet.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::piggyback.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::cic.tightness::labeled::k_grad_median --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.open.O1::linear::wild_fp_rate regenerated/numbers.json::malnet.open.O1::linear::wild_fp_rate regenerated/numbers.json::piggyback.open.O1::linear::wild_fp_rate regenerated/numbers.json::cic.open.O1::linear::wild_fp_rate --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.open.O1::linear::wild_fp_rate regenerated/numbers.json::malnet.open.O1::linear::wild_fp_rate regenerated/numbers.json::piggyback.open.O1::linear::wild_fp_rate regenerated/numbers.json::cic.open.O1::linear::wild_fp_rate --percent --digits 0

# 10-page version: Introduction
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.n --sep

# 10-page version: Corpora
run cat regenerated/tables_10p/corpora.tex
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::apps --sep
run "$PY" scripts/query.py count data/manifests/proxy.csv --where closed_world=1 --sep
run "$PY" scripts/query.py group-size data/manifests/proxy.csv --by label --where closed_world=1 --largest-share 2 --percent --digits 0
run "$PY" scripts/query.py count data/manifests/proxy.csv --where closed_world=0 --sep
run "$PY" scripts/query.py count data/manifests/wild_play.csv --where proxy_marker_found=0 --sep
run "$PY" scripts/query.py count data/manifests/malnet.csv --where closed_world=1 --distinct label
run "$PY" scripts/query.py group-size data/manifests/malnet.csv --by label --where closed_world=1 --over-groups max
run "$PY" scripts/query.py count data/manifests/malnet.csv --where closed_world=0
run "$PY" scripts/query.py count data/manifests/cic_andmal2020.csv --where closed_world=0

# 10-page version: Method, evaluation protocol
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/e2a_standardized.json::labeled::dim --sep
run "$PY" scripts/query.py count data/manifests/proxy.csv --where closed_world=1 --distinct lineage_group_closed
run "$PY" scripts/query.py count data/manifests/proxy.csv --where closed_world=1 --distinct dex_group --sep
run "$PY" scripts/query.py group-size 'data/results/datasets/proxy_v3_gneardup_fcg/tier4/p3/frontier.json::sgd_*::rows' 'data/results/datasets/malnet/tier4/p3/frontier.json::sgd_*::rows' 'data/results/datasets/piggyback/tier4/p3/frontier.json::sgd_*::rows' 'data/results/datasets/cic_andmal/tier4/p3/frontier.json::sgd_*::rows' --by family --distinct sha --over-groups max

# 10-page version: Results
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/e2a_standardized.json::labeled::dim --sep

# 10-page version: Results, RQ1 linearity
run cat regenerated/tables_10p/linearity.tex
run "$PY" scripts/query.py diff-stat max regenerated/numbers.json::proxy.ladder.identifier.best::f1 regenerated/numbers.json::proxy.ladder.identifier.linear regenerated/numbers.json::malnet.ladder.identifier.best::f1 regenerated/numbers.json::malnet.ladder.identifier.linear regenerated/numbers.json::piggyback.ladder.identifier.best::f1 regenerated/numbers.json::piggyback.ladder.identifier.linear regenerated/numbers.json::cic.ladder.identifier.best::f1 regenerated/numbers.json::cic.ladder.identifier.linear --digits 2
run "$PY" scripts/query.py stat max 'data/results/datasets/piggyback/tier4/l1_ladder_summary.json::fuzzy::*::diff_vs_linear' 'data/results/datasets/piggyback/tier4/l1_ladder_summary.json::api::*::diff_vs_linear' 'data/results/datasets/piggyback/tier4/l1_ladder_summary.json::structure::*::diff_vs_linear' 'data/results/datasets/cic_andmal/tier4/l1_ladder_summary.json::fuzzy::*::diff_vs_linear' 'data/results/datasets/cic_andmal/tier4/l1_ladder_summary.json::api::*::diff_vs_linear' 'data/results/datasets/cic_andmal/tier4/l1_ladder_summary.json::structure::*::diff_vs_linear' --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.hardmargin_err.identifier --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.hardmargin_err.identifier --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.renaming.flip_rate regenerated/numbers.json::malnet.renaming.flip_rate regenerated/numbers.json::piggyback.renaming.flip_rate regenerated/numbers.json::cic.renaming.flip_rate --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.renaming.flip_rate regenerated/numbers.json::malnet.renaming.flip_rate regenerated/numbers.json::piggyback.renaming.flip_rate regenerated/numbers.json::cic.renaming.flip_rate --percent --digits 0

# 10-page version: Results, RQ2 attribution and localization
run cat regenerated/tables_10p/localization.tex
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.ident.median_signed_error_max regenerated/numbers.json::malnet.ident.median_signed_error_max regenerated/numbers.json::piggyback.ident.median_signed_error_max regenerated/numbers.json::cic.ident.median_signed_error_max --digits 3
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.ident.colouring_M::fuzzy::ratio_to_labeled regenerated/numbers.json::malnet.ident.colouring_M::fuzzy::ratio_to_labeled regenerated/numbers.json::piggyback.ident.colouring_M::fuzzy::ratio_to_labeled regenerated/numbers.json::cic.ident.colouring_M::fuzzy::ratio_to_labeled --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.ident.colouring_M::fuzzy::ratio_to_labeled regenerated/numbers.json::malnet.ident.colouring_M::fuzzy::ratio_to_labeled regenerated/numbers.json::piggyback.ident.colouring_M::fuzzy::ratio_to_labeled regenerated/numbers.json::cic.ident.colouring_M::fuzzy::ratio_to_labeled --digits 2
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.ident.colouring_M::api::ratio_to_labeled regenerated/numbers.json::malnet.ident.colouring_M::api::ratio_to_labeled regenerated/numbers.json::piggyback.ident.colouring_M::api::ratio_to_labeled regenerated/numbers.json::cic.ident.colouring_M::api::ratio_to_labeled --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.ident.colouring_M::api::ratio_to_labeled regenerated/numbers.json::malnet.ident.colouring_M::api::ratio_to_labeled regenerated/numbers.json::piggyback.ident.colouring_M::api::ratio_to_labeled regenerated/numbers.json::cic.ident.colouring_M::api::ratio_to_labeled --digits 2
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/e2a_standardized.json::labeled::dim --sep
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.loc1024.ratio_by_explainer::lin_exact --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.loc1024.ratio_by_explainer::lin_exact --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.loc1024.ratio_by_explainer::lin_exact --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.loc1024.ratio_by_explainer::*' 'regenerated/numbers.json::malnet.loc1024.ratio_by_explainer::*' 'regenerated/numbers.json::piggyback.loc1024.ratio_by_explainer::*' --skip-key lin_exact --digits 2
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.loc1024.ratio_by_explainer::*' 'regenerated/numbers.json::malnet.loc1024.ratio_by_explainer::*' 'regenerated/numbers.json::piggyback.loc1024.ratio_by_explainer::*' --skip-key lin_exact --digits 2
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.mx::lin_exact::M3_flip_0.01 regenerated/numbers.json::malnet.mx::lin_exact::M3_flip_0.01 regenerated/numbers.json::piggyback.mx::lin_exact::M3_flip_0.01 --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.mx::lin_exact::M3_flip_0.01 regenerated/numbers.json::malnet.mx::lin_exact::M3_flip_0.01 regenerated/numbers.json::piggyback.mx::lin_exact::M3_flip_0.01 --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.mx::lin_exact::M3_retention_random_0.01 regenerated/numbers.json::malnet.mx::lin_exact::M3_retention_random_0.01 regenerated/numbers.json::piggyback.mx::lin_exact::M3_retention_random_0.01 --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.mx::lin_exact::M3_retention_random_0.01 regenerated/numbers.json::malnet.mx::lin_exact::M3_retention_random_0.01 regenerated/numbers.json::piggyback.mx::lin_exact::M3_retention_random_0.01 --digits 2
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.mx::mlp_ig::M3_flip_0.01 regenerated/numbers.json::malnet.mx::mlp_ig::M3_flip_0.01 regenerated/numbers.json::piggyback.mx::mlp_ig::M3_flip_0.01 regenerated/numbers.json::proxy.mx::mlp_gxi::M3_flip_0.01 regenerated/numbers.json::malnet.mx::mlp_gxi::M3_flip_0.01 regenerated/numbers.json::piggyback.mx::mlp_gxi::M3_flip_0.01 --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.mx::mlp_ig::M3_flip_0.01 regenerated/numbers.json::malnet.mx::mlp_ig::M3_flip_0.01 regenerated/numbers.json::piggyback.mx::mlp_ig::M3_flip_0.01 regenerated/numbers.json::proxy.mx::mlp_gxi::M3_flip_0.01 regenerated/numbers.json::malnet.mx::mlp_gxi::M3_flip_0.01 regenerated/numbers.json::piggyback.mx::mlp_gxi::M3_flip_0.01 --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.mx::lin_exact::M4_top100_jaccard --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.mx::lin_exact::M4_top100_jaccard --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.mx::lin_exact::M4_top100_jaccard --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.mx::rf_path::M4_top100_jaccard --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.seed_spearman::median_spearman_across_seeds --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.seed_spearman::median_spearman_across_seeds --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.seed_spearman::median_spearman_across_seeds --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.seed_spearman::p10 --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.gnn::ratio --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.gnn::macro_f1 --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.highdim.ratio_by_dim::1048576 --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.highdim.ratio_by_dim::1048576 --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.highdim.ratio_by_dim::1048576 --digits 1
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_loc::cfg|identifier|clean::1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_loc::cfg|identifier|clean::1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_loc::cfg|identifier|clean::1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_loc::concat|identifier|clean::1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_loc::concat|identifier|clean::1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_loc::concat|identifier|clean::1048576::ratio' --digits 2
run "$PY" scripts/query.py stat max 'data/results/datasets/proxy_v3/exp_tier1_inflation.json::k_sweep::*' --sep
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/e2a_standardized.json::labeled::dim --sep
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.infl::still_correct regenerated/numbers.json::malnet.infl::still_correct regenerated/numbers.json::piggyback.infl::still_correct --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.infl::still_correct regenerated/numbers.json::malnet.infl::still_correct regenerated/numbers.json::piggyback.infl::still_correct --percent --digits 0
run "$PY" scripts/query.py diff-stat max regenerated/numbers.json::proxy.infl::iso0::1048576 regenerated/numbers.json::proxy.infl::isok::1048576 regenerated/numbers.json::malnet.infl::iso0::1048576 regenerated/numbers.json::malnet.infl::isok::1048576 regenerated/numbers.json::piggyback.infl::iso0::1048576 regenerated/numbers.json::piggyback.infl::isok::1048576 --percent --digits 0 --rounding ceiling
run "$PY" scripts/query.py diff-stat max regenerated/numbers.json::proxy.infl::iso0::1048576 regenerated/numbers.json::proxy.infl::isok::1048576 regenerated/numbers.json::malnet.infl::iso0::1048576 regenerated/numbers.json::malnet.infl::isok::1048576 regenerated/numbers.json::piggyback.infl::iso0::1048576 regenerated/numbers.json::piggyback.infl::isok::1048576 --percent --digits 1
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.colour::fuzzy|clean|1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.colour::fuzzy|clean|1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.colour::fuzzy|clean|1048576::ratio' --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.granularity::fuzzy::*' 'regenerated/numbers.json::malnet.granularity::fuzzy::*' 'regenerated/numbers.json::piggyback.granularity::fuzzy::*' --percent --digits 0
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.granularity::fuzzy::*' 'regenerated/numbers.json::malnet.granularity::fuzzy::*' 'regenerated/numbers.json::piggyback.granularity::fuzzy::*' --percent --digits 0
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.granularity::api::*' 'regenerated/numbers.json::malnet.granularity::api::*' 'regenerated/numbers.json::piggyback.granularity::api::*' --percent --digits 0
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.granularity::api::*' 'regenerated/numbers.json::malnet.granularity::api::*' 'regenerated/numbers.json::piggyback.granularity::api::*' --percent --digits 0
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.granularity::structure::*' 'regenerated/numbers.json::malnet.granularity::structure::*' 'regenerated/numbers.json::piggyback.granularity::structure::*' --percent --digits 0
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.granularity::structure::*' 'regenerated/numbers.json::malnet.granularity::structure::*' 'regenerated/numbers.json::piggyback.granularity::structure::*' --percent --digits 0
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::malnet.transplant::token_frac::*' --percent --digits 0
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::malnet.transplant::token_frac::*' --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.transplant::A2c::unseen::follows_sdk --percent --digits 0
run "$PY" scripts/query.py nth-largest 3 'regenerated/numbers.json::proxy.transplant::token_frac::*' --percent --digits 0 --rounding ceiling
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.transplant::A2a::sdk --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.transplant::A2a::full --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.transplant::A2c::unseen::follows_sdk --percent --digits 0
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.transplant::A2b::*::sdk' --digits 2
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.transplant::A2b::*::sdk' --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::malnet.transplant::A2b::*::sdk' --digits 2
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::malnet.transplant::A2b::*::sdk' --digits 2

# 10-page version: Results, RQ3 certified robustness
run cat regenerated/tables_10p/robustness.tex
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/e2a_standardized.json::labeled::dim --sep
run "$PY" scripts/query.py above-one min regenerated/numbers.json::proxy.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::malnet.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::piggyback.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::cic.tightness::labeled::ceiling_over_floor_median --percent --digits 0
run "$PY" scripts/query.py above-one max regenerated/numbers.json::proxy.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::malnet.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::piggyback.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::cic.tightness::labeled::ceiling_over_floor_median --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.salt_spearman regenerated/numbers.json::malnet.salt_spearman regenerated/numbers.json::piggyback.salt_spearman regenerated/numbers.json::cic.salt_spearman --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.salt_spearman regenerated/numbers.json::malnet.salt_spearman regenerated/numbers.json::piggyback.salt_spearman regenerated/numbers.json::cic.salt_spearman --digits 2
run "$PY" scripts/query.py ratio-stat min regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::malnet.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::piggyback.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::cic.tightness::labeled::k_grad_median --digits 0
run "$PY" scripts/query.py ratio-stat max regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::malnet.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::piggyback.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::cic.tightness::labeled::k_grad_median --digits 0
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.thm_bounds::summary::labeled|*::log10_slack_global_median' 'regenerated/numbers.json::malnet.thm_bounds::summary::labeled|*::log10_slack_global_median' 'regenerated/numbers.json::piggyback.thm_bounds::summary::labeled|*::log10_slack_global_median' 'regenerated/numbers.json::cic.thm_bounds::summary::labeled|*::log10_slack_global_median' --digits 1
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.thm_bounds::summary::labeled|*::log10_slack_global_median' 'regenerated/numbers.json::malnet.thm_bounds::summary::labeled|*::log10_slack_global_median' 'regenerated/numbers.json::piggyback.thm_bounds::summary::labeled|*::log10_slack_global_median' 'regenerated/numbers.json::cic.thm_bounds::summary::labeled|*::log10_slack_global_median' --digits 1
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.r4::flip_*' 'regenerated/numbers.json::malnet.r4::flip_*' 'regenerated/numbers.json::piggyback.r4::flip_*' 'regenerated/numbers.json::cic.r4::flip_*' --percent --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.steer_graph::never_flipped_frac --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.steer_graph::never_flipped_frac --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.steer_graph::never_flipped_frac --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.steer_graph::never_flipped_frac --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.realism::labeled::greybox_disjoint regenerated/numbers.json::malnet.realism::labeled::greybox_disjoint regenerated/numbers.json::piggyback.realism::labeled::greybox_disjoint regenerated/numbers.json::cic.realism::labeled::greybox_disjoint --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.realism::labeled::greybox_disjoint regenerated/numbers.json::malnet.realism::labeled::greybox_disjoint regenerated/numbers.json::piggyback.realism::labeled::greybox_disjoint regenerated/numbers.json::cic.realism::labeled::greybox_disjoint --percent --digits 0
run "$PY" scripts/query.py value data/results/datasets/malnet/tier4/v3/attack_realism.json::budget
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.realism::labeled::transfer_reseed regenerated/numbers.json::malnet.realism::labeled::transfer_reseed regenerated/numbers.json::piggyback.realism::labeled::transfer_reseed regenerated/numbers.json::cic.realism::labeled::transfer_reseed --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.realism::labeled::transfer_reseed regenerated/numbers.json::malnet.realism::labeled::transfer_reseed regenerated/numbers.json::piggyback.realism::labeled::transfer_reseed regenerated/numbers.json::cic.realism::labeled::transfer_reseed --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.bundle::greedy::success --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.bundle::whole_app_flip_rate_mean --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::malnet.bundle::whole_app_flip_rate_mean regenerated/numbers.json::piggyback.bundle::whole_app_flip_rate_mean regenerated/numbers.json::cic.bundle::whole_app_flip_rate_mean --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::malnet.bundle::whole_app_flip_rate_mean regenerated/numbers.json::piggyback.bundle::whole_app_flip_rate_mean regenerated/numbers.json::cic.bundle::whole_app_flip_rate_mean --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.frontier.combined::k_steer_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.frontier.combined::k_steer_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.frontier.identifier::k_steer_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.frontier.identifier::k_steer_median --digits 0
run "$PY" scripts/query.py ratio-stat min regenerated/numbers.json::piggyback.frontier.combined::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::piggyback.frontier.fuzzy::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::piggyback.frontier.api::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::piggyback.frontier.degree::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.combined::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.fuzzy::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.api::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.degree::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median --digits 1
run "$PY" scripts/query.py ratio-stat max regenerated/numbers.json::piggyback.frontier.combined::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::piggyback.frontier.fuzzy::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::piggyback.frontier.api::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::piggyback.frontier.degree::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.combined::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.fuzzy::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.api::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.degree::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median --digits 1
run "$PY" scripts/query.py ratio-stat min regenerated/numbers.json::proxy.linf::0.25::k_steer_median regenerated/numbers.json::proxy.linf::1.0::k_steer_median regenerated/numbers.json::malnet.linf::0.25::k_steer_median regenerated/numbers.json::malnet.linf::1.0::k_steer_median regenerated/numbers.json::piggyback.linf::0.25::k_steer_median regenerated/numbers.json::piggyback.linf::1.0::k_steer_median regenerated/numbers.json::cic.linf::0.25::k_steer_median regenerated/numbers.json::cic.linf::1.0::k_steer_median --digits 1
run "$PY" scripts/query.py ratio-stat max regenerated/numbers.json::proxy.linf::0.25::k_steer_median regenerated/numbers.json::proxy.linf::1.0::k_steer_median regenerated/numbers.json::malnet.linf::0.25::k_steer_median regenerated/numbers.json::malnet.linf::1.0::k_steer_median regenerated/numbers.json::piggyback.linf::0.25::k_steer_median regenerated/numbers.json::piggyback.linf::1.0::k_steer_median regenerated/numbers.json::cic.linf::0.25::k_steer_median regenerated/numbers.json::cic.linf::1.0::k_steer_median --digits 1
run "$PY" scripts/query.py diff-stat max regenerated/numbers.json::proxy.linf::1.0::f1 regenerated/numbers.json::proxy.linf::0.25::f1 regenerated/numbers.json::malnet.linf::1.0::f1 regenerated/numbers.json::malnet.linf::0.25::f1 regenerated/numbers.json::piggyback.linf::1.0::f1 regenerated/numbers.json::piggyback.linf::0.25::f1 regenerated/numbers.json::cic.linf::1.0::f1 regenerated/numbers.json::cic.linf::0.25::f1 --digits 2
run "$PY" scripts/query.py ratio-stat min 'regenerated/numbers.json::proxy.cfg_cert::cfg|identifier::k_certgrad_median' 'regenerated/numbers.json::proxy.cfg_cert::fcg_block|identifier::k_certgrad_median' 'regenerated/numbers.json::malnet.cfg_cert::cfg|identifier::k_certgrad_median' 'regenerated/numbers.json::malnet.cfg_cert::fcg_block|identifier::k_certgrad_median' 'regenerated/numbers.json::piggyback.cfg_cert::cfg|identifier::k_certgrad_median' 'regenerated/numbers.json::piggyback.cfg_cert::fcg_block|identifier::k_certgrad_median' 'regenerated/numbers.json::cic.cfg_cert::cfg|identifier::k_certgrad_median' 'regenerated/numbers.json::cic.cfg_cert::fcg_block|identifier::k_certgrad_median' --digits 1
run "$PY" scripts/query.py ratio-stat max 'regenerated/numbers.json::proxy.cfg_cert::cfg|identifier::k_certgrad_median' 'regenerated/numbers.json::proxy.cfg_cert::fcg_block|identifier::k_certgrad_median' 'regenerated/numbers.json::malnet.cfg_cert::cfg|identifier::k_certgrad_median' 'regenerated/numbers.json::malnet.cfg_cert::fcg_block|identifier::k_certgrad_median' 'regenerated/numbers.json::piggyback.cfg_cert::cfg|identifier::k_certgrad_median' 'regenerated/numbers.json::piggyback.cfg_cert::fcg_block|identifier::k_certgrad_median' 'regenerated/numbers.json::cic.cfg_cert::cfg|identifier::k_certgrad_median' 'regenerated/numbers.json::cic.cfg_cert::fcg_block|identifier::k_certgrad_median' --digits 1
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.tree_cert::k_attack_median regenerated/numbers.json::malnet.tree_cert::k_attack_median regenerated/numbers.json::piggyback.tree_cert::k_attack_median regenerated/numbers.json::cic.tree_cert::k_attack_median --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.tree_cert::k_attack_median regenerated/numbers.json::malnet.tree_cert::k_attack_median regenerated/numbers.json::piggyback.tree_cert::k_attack_median regenerated/numbers.json::cic.tree_cert::k_attack_median --digits 0

# 10-page version: Results, the open world and time
run cat regenerated/tables_10p/openworld.tex
run "$PY" scripts/query.py count data/manifests/wild_play.csv --where proxy_marker_found=0 --where fcg_vector=1 --sep
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.open.O1::linear::wild_fp_rate regenerated/numbers.json::malnet.open.O1::linear::wild_fp_rate regenerated/numbers.json::piggyback.open.O1::linear::wild_fp_rate regenerated/numbers.json::cic.open.O1::linear::wild_fp_rate --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.open.O1::linear::wild_fp_rate regenerated/numbers.json::malnet.open.O1::linear::wild_fp_rate regenerated/numbers.json::piggyback.open.O1::linear::wild_fp_rate regenerated/numbers.json::cic.open.O1::linear::wild_fp_rate --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.open.abstain::auroc_proxy_vs_wild regenerated/numbers.json::malnet.open.abstain::auroc_proxy_vs_wild regenerated/numbers.json::cic.open.abstain::auroc_proxy_vs_wild --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.open.abstain::auroc_proxy_vs_wild regenerated/numbers.json::malnet.open.abstain::auroc_proxy_vs_wild regenerated/numbers.json::cic.open.abstain::auroc_proxy_vs_wild --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.openset_dex_msp::auroc --digits 3
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.wild_by_coloring::combined::0 regenerated/numbers.json::proxy.wild_by_coloring::fuzzy::0 regenerated/numbers.json::proxy.wild_by_coloring::api::0 regenerated/numbers.json::proxy.wild_by_coloring::structure::0 --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.wild_by_coloring::combined::0 regenerated/numbers.json::proxy.wild_by_coloring::fuzzy::0 regenerated/numbers.json::proxy.wild_by_coloring::api::0 regenerated/numbers.json::proxy.wild_by_coloring::structure::0 --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.temporal::gap regenerated/numbers.json::malnet.temporal::gap regenerated/numbers.json::piggyback.temporal::gap regenerated/numbers.json::cic.temporal::gap --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.temporal::gap regenerated/numbers.json::malnet.temporal::gap regenerated/numbers.json::piggyback.temporal::gap regenerated/numbers.json::cic.temporal::gap --digits 2

# 10-page version: Results, RQ4 separability
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.rq4::k_norm --permille --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.rq4::k_norm --permille --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.rq4::k_norm --permille --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::apps::min_cut::n
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::apps::min_cut::spearman_with_k_norm --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::apps::rho::spearman_with_k_norm --digits 2

# 10-page version: Discussion
run "$PY" scripts/query.py group-size 'data/results/datasets/proxy_v3_gneardup_fcg/tier4/p3/frontier.json::sgd_*::rows' 'data/results/datasets/malnet/tier4/p3/frontier.json::sgd_*::rows' 'data/results/datasets/piggyback/tier4/p3/frontier.json::sgd_*::rows' 'data/results/datasets/cic_andmal/tier4/p3/frontier.json::sgd_*::rows' --by family --distinct sha --over-groups max

# extended version: Abstract
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.n --sep
run "$PY" scripts/query.py count data/manifests/malnet.csv --where closed_world=1 --distinct label
run "$PY" scripts/query.py above-one min regenerated/numbers.json::proxy.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::malnet.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::piggyback.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::cic.tightness::labeled::ceiling_over_floor_median --percent --digits 0
run "$PY" scripts/query.py above-one max regenerated/numbers.json::proxy.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::malnet.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::piggyback.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::cic.tightness::labeled::ceiling_over_floor_median --percent --digits 0
run "$PY" scripts/query.py ratio-stat min regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::malnet.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::piggyback.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::cic.tightness::labeled::k_grad_median --digits 0
run "$PY" scripts/query.py ratio-stat max regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::malnet.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::piggyback.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::cic.tightness::labeled::k_grad_median --digits 0

# extended version: Introduction, research questions
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.n --sep

# extended version: Dataset, corpus
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::apps --sep
run "$PY" scripts/query.py largest-share 2 'data/results/extra/dex_reuse_v3.json::per_family::*::apps' --percent --digits 0
run "$PY" scripts/query.py group-size 'data/results/datasets/proxy_v3_gneardup_fcg/tier4/x2_method_level.json::apps::*' --by family --over-groups max
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::per_family::traffmonetizer::apps
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::per_family::ipninja::apps
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::per_family::cashraven::apps
run "$PY" scripts/query.py count data/manifests/proxy.csv --where closed_world=1 --sep
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::per_family::cashraven::apps --sep
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::per_family::ipidea::apps --sep
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::per_family::ipninja::apps --sep
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::per_family::luminati::apps --sep
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::per_family::monkeysocks::apps --sep
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::per_family::oxylabs::apps --sep
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::per_family::popa::apps --sep
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::per_family::traffmonetizer::apps --sep
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::apps --sep
run "$PY" scripts/query.py count data/manifests/proxy.csv --where closed_world=0 --sep
run "$PY" scripts/query.py count data/manifests/wild_play.csv --sep
run "$PY" scripts/query.py count data/manifests/wild_play.csv --where proxy_marker_found=1
run "$PY" scripts/query.py count data/manifests/wild_play.csv --where proxy_marker_found=0 --sep

# extended version: Dataset, comparison corpora
run "$PY" scripts/query.py count data/manifests/malnet.csv --where closed_world=1 --distinct label
run "$PY" scripts/query.py group-size data/manifests/malnet.csv --by label --where closed_world=1 --over-groups max
run "$PY" scripts/query.py count data/manifests/malnet.csv --where closed_world=0
run "$PY" scripts/query.py count data/manifests/malnet.csv --sep
run "$PY" scripts/query.py group-stat median 'data/results/datasets/malnet_payload/tier4/x2_method_level.json::apps::*' --by family --field lin_exact::n_target --digits -2 --rounding half-even --sep --over-groups min
run "$PY" scripts/query.py group-stat median 'data/results/datasets/malnet_payload/tier4/x2_method_level.json::apps::*' --by family --field lin_exact::n_target --digits -2 --rounding half-even --sep --over-groups max
run "$PY" scripts/query.py count data/manifests/proxy.csv --where closed_world=1 --distinct lineage_group_closed --sep
run "$PY" scripts/query.py count data/manifests/proxy.csv --where closed_world=1 --distinct dex_group --sep
run "$PY" scripts/query.py count data/manifests/proxy.csv --where closed_world=1 --sep

# extended version: Dataset, graph extraction
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::apps --sep
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::dex_occurrences --sep
run "$PY" scripts/query.py value data/results/extra/dex_reuse_v3.json::dex_distinct --sep
run "$PY" scripts/query.py stat min 'data/results/extra/mincut_fcg_nodes.json::per_provider::*::fcg_nodes_median' --digits 0 --sep
run "$PY" scripts/query.py stat max 'data/results/extra/mincut_fcg_nodes.json::per_provider::*::fcg_nodes_median' --digits 0 --sep
run "$PY" scripts/query.py value data/results/extra/cfg_blocks.json::proxy_v3::median_nonlibrary_blocks --digits -3 --rounding half-even --sep
run "$PY" scripts/query.py stat min data/results/extra/cfg_blocks.json::malnet::median_nonlibrary_blocks data/results/extra/cfg_blocks.json::piggyback::median_nonlibrary_blocks data/results/extra/cfg_blocks.json::cic_andmal::median_nonlibrary_blocks --digits -2 --rounding half-even --sep
run "$PY" scripts/query.py stat max data/results/extra/cfg_blocks.json::malnet::median_nonlibrary_blocks data/results/extra/cfg_blocks.json::piggyback::median_nonlibrary_blocks data/results/extra/cfg_blocks.json::cic_andmal::median_nonlibrary_blocks --digits -2 --rounding half-even --sep
run "$PY" scripts/query.py stat min 'data/results/extra/cfg_blocks.json::*::median_mean_blocks_per_method' --digits 1
run "$PY" scripts/query.py stat max 'data/results/extra/cfg_blocks.json::*::median_mean_blocks_per_method' --digits 1

# extended version: Dataset, evaluation protocol and subsets
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/e2a_standardized.json::labeled::dim --sep
run "$PY" scripts/query.py group-size 'data/results/datasets/proxy_v3_gneardup_fcg/tier4/x2_method_level.json::apps::*' --by family --over-groups max
run "$PY" scripts/query.py group-size 'data/results/datasets/malnet_payload/tier4/x2_method_level.json::apps::*' --by family --over-groups max
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.seed_spearman::n
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.seed_spearman::n
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.seed_spearman::n
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::apps_analysed
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::apps_unambiguous

# extended version: Dataset, structural characteristics
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::cashraven::fcg_nodes_median --digits 0 --sep
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::cashraven::sdk_methods_median --digits 0 --sep
run "$PY" scripts/query.py field-stat median 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=cashraven
run "$PY" scripts/query.py field-stat max 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=cashraven
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::ipidea::fcg_nodes_median --digits 0 --sep
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::ipidea::sdk_methods_median --digits 0 --sep
run "$PY" scripts/query.py field-stat median 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=ipidea
run "$PY" scripts/query.py field-stat max 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=ipidea
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::ipninja::fcg_nodes_median --digits 0 --sep
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::ipninja::sdk_methods_median --digits 0 --sep
run "$PY" scripts/query.py field-stat median 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=ipninja
run "$PY" scripts/query.py field-stat max 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=ipninja
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::luminati::fcg_nodes_median --digits 0 --sep
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::luminati::sdk_methods_median --digits 0 --sep
run "$PY" scripts/query.py field-stat median 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=luminati
run "$PY" scripts/query.py field-stat max 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=luminati
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::monkeysocks::fcg_nodes_median --digits 0 --sep
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::monkeysocks::sdk_methods_median --digits 0 --sep
run "$PY" scripts/query.py field-stat median 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=monkeysocks
run "$PY" scripts/query.py field-stat max 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=monkeysocks
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::oxylabs::fcg_nodes_median --digits 0 --sep
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::oxylabs::sdk_methods_median --digits 0 --sep
run "$PY" scripts/query.py field-stat median 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=oxylabs
run "$PY" scripts/query.py field-stat max 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=oxylabs
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::popa::fcg_nodes_median --digits 0 --sep
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::popa::sdk_methods_median --digits 0 --sep
run "$PY" scripts/query.py field-stat median 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=popa
run "$PY" scripts/query.py field-stat max 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=popa
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::traffmonetizer::fcg_nodes_median --digits 0 --sep
run "$PY" scripts/query.py value data/results/extra/mincut_fcg_nodes.json::per_provider::traffmonetizer::sdk_methods_median --digits 0 --sep
run "$PY" scripts/query.py field-stat median 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=traffmonetizer
run "$PY" scripts/query.py field-stat max 'data/results/extra/mincut_v3.json::apps::*' --field cut --where family=traffmonetizer
run "$PY" scripts/query.py field-stat count 'data/results/extra/mincut_v3.json::apps::*' --field cut
run "$PY" scripts/query.py group-stat median 'data/results/extra/mincut_v3.json::apps::*' --by family --field cut --digits 0 --over-groups min
run "$PY" scripts/query.py group-stat median 'data/results/extra/mincut_v3.json::apps::*' --by family --field cut --digits 0 --over-groups max
run "$PY" scripts/query.py stat min 'data/results/extra/mincut_fcg_nodes.json::per_provider::*::sdk_methods_median' --digits 0 --sep
run "$PY" scripts/query.py stat max 'data/results/extra/mincut_fcg_nodes.json::per_provider::*::sdk_methods_median' --digits 0 --sep

# extended version: Methodology
run "$PY" scripts/query.py count data/manifests/proxy.csv --where closed_world=0 --sep
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/e2a_standardized.json::labeled::dim --sep
run "$PY" scripts/query.py diff-stat max 'regenerated/numbers.json::*.h.fuzzy::1' 'regenerated/numbers.json::*.h.fuzzy::0' 'regenerated/numbers.json::*.h.api::1' 'regenerated/numbers.json::*.h.api::0' 'regenerated/numbers.json::*.h.degree::1' 'regenerated/numbers.json::*.h.degree::0' --digits 2
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/e2a_standardized.json::labeled::dim --sep
run "$PY" scripts/query.py count data/manifests/malnet.csv --where closed_world=1 --distinct label
run "$PY" scripts/query.py group-size 'data/results/datasets/malnet_payload/tier4/x2_method_level.json::apps::*' --by family --over-groups max
run "$PY" scripts/query.py value data/results/datasets/malnet/tier4/v3/bundle_attack.json::n_donors

# extended version: Results
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/e2a_standardized.json::labeled::dim --sep

# extended version: Results, linearity of the attribution problem
run cat regenerated/tables_extended/rq1_linearity.tex
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/e2a_standardized.json::labeled::dim --sep
run "$PY" scripts/query.py diff-stat max regenerated/numbers.json::proxy.ladder.identifier.best::f1 regenerated/numbers.json::proxy.ladder.identifier.linear regenerated/numbers.json::malnet.ladder.identifier.best::f1 regenerated/numbers.json::malnet.ladder.identifier.linear regenerated/numbers.json::piggyback.ladder.identifier.best::f1 regenerated/numbers.json::piggyback.ladder.identifier.linear regenerated/numbers.json::cic.ladder.identifier.best::f1 regenerated/numbers.json::cic.ladder.identifier.linear --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.ladder.identifier.best::f1 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.ladder.identifier.linear --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.ladder.identifier.best::p --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.ladder.identifier.best::f1 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.ladder.identifier.linear --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.ladder.identifier.best::p --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.ladder.identifier.best::f1 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.ladder.identifier.linear --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::cic.ladder.identifier.best::f1 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::cic.ladder.identifier.linear --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.ladder.fuzzy.best::f1 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.ladder.fuzzy.linear --digits 3
run "$PY" scripts/query.py stat max 'data/results/datasets/piggyback/tier4/l1_ladder_summary.json::fuzzy::*::diff_vs_linear' 'data/results/datasets/piggyback/tier4/l1_ladder_summary.json::api::*::diff_vs_linear' 'data/results/datasets/piggyback/tier4/l1_ladder_summary.json::structure::*::diff_vs_linear' 'data/results/datasets/cic_andmal/tier4/l1_ladder_summary.json::fuzzy::*::diff_vs_linear' 'data/results/datasets/cic_andmal/tier4/l1_ladder_summary.json::api::*::diff_vs_linear' 'data/results/datasets/cic_andmal/tier4/l1_ladder_summary.json::structure::*::diff_vs_linear' --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::cic.hardmargin_err.identifier --percent --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.hardmargin_err.identifier --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.hardmargin_err.identifier --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.cka.labeled --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.cka.labeled --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::cic.cka.labeled --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.cka.labeled --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.margin.labeled::p10 --digits 2
run "$PY" scripts/query.py stat min regenerated/numbers.json::malnet.margin.labeled::p10 regenerated/numbers.json::piggyback.margin.labeled::p10 regenerated/numbers.json::cic.margin.labeled::p10 --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::malnet.margin.labeled::p10 regenerated/numbers.json::piggyback.margin.labeled::p10 regenerated/numbers.json::cic.margin.labeled::p10 --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.synthetic::additive::*' --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.synthetic::xor::naive_bayes --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.synthetic::xor::logistic --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.synthetic::xor::forest --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.nb_vs_logistic.identifier::0 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.nb_vs_logistic.identifier::1 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::cic.nb_vs_logistic.identifier::0 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::cic.nb_vs_logistic.identifier::1 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.nb_vs_logistic.identifier::0 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.nb_vs_logistic.identifier::1 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.nb_vs_logistic.identifier::0 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.nb_vs_logistic.identifier::1 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.dim_scaling::identifier::1024::linear --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.dim_scaling::identifier::1024::forest --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.dim_scaling::identifier::65536::linear --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.dim_scaling::identifier::65536::forest --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.dim_scaling::identifier::1024::linear --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.dim_scaling::identifier::1024::forest --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.dim_scaling::identifier::65536::linear --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.dim_scaling::identifier::65536::forest --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.dim_scaling::degree::65536::linear --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.dim_scaling::degree::65536::forest --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.dim_scaling::degree::65536::linear --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.dim_scaling::degree::65536::forest --digits 2
run cat regenerated/tables_extended/rq1_representations.tex
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.e2a.labeled::raw --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.e2a.labeled::std --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.e2a.labeled::raw --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.e2a.labeled::std --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.e2a.structure::raw --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.e2a.structure::std --digits 3
run "$PY" scripts/query.py diff regenerated/numbers.json::proxy.e2a.combined::std regenerated/numbers.json::proxy.e2a.combined::raw --digits 2
run "$PY" scripts/query.py diff regenerated/numbers.json::proxy.e2a.fuzzy::std regenerated/numbers.json::proxy.e2a.fuzzy::raw --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.e2a.mama_package::raw --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.e2a.mama_package::std --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.ladder.combined.linear --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.ladder.identifier.linear --digits 3

# extended version: Results, initial colorings
run cat regenerated/tables_extended/rq1_refinement.tex
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.h.identifier::0 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.h.identifier::1 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::cic.h.combined::0 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::cic.h.combined::1 --digits 3
run "$PY" scripts/query.py diff-stat min regenerated/numbers.json::proxy.h.degree::1 regenerated/numbers.json::proxy.h.degree::0 regenerated/numbers.json::malnet.h.degree::1 regenerated/numbers.json::malnet.h.degree::0 regenerated/numbers.json::piggyback.h.degree::1 regenerated/numbers.json::piggyback.h.degree::0 regenerated/numbers.json::cic.h.degree::1 regenerated/numbers.json::cic.h.degree::0 --digits 2
run "$PY" scripts/query.py diff-stat max regenerated/numbers.json::proxy.h.degree::1 regenerated/numbers.json::proxy.h.degree::0 regenerated/numbers.json::malnet.h.degree::1 regenerated/numbers.json::malnet.h.degree::0 regenerated/numbers.json::piggyback.h.degree::1 regenerated/numbers.json::piggyback.h.degree::0 regenerated/numbers.json::cic.h.degree::1 regenerated/numbers.json::cic.h.degree::0 --digits 2
run "$PY" scripts/query.py diff regenerated/numbers.json::proxy.h.api::1 regenerated/numbers.json::proxy.h.api::0 --digits 2
run "$PY" scripts/query.py diff regenerated/numbers.json::piggyback.h.api::0 regenerated/numbers.json::piggyback.h.api::1 --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.renaming.flip_rate --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.renaming.flip_rate --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.renaming.flip_rate --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.renaming.flip_rate --percent --digits 0
run "$PY" scripts/query.py value data/results/datasets/proxy_v3/exp_tier1_seed_mi.json::modes::combined::J_maximin_bits --digits 2
run "$PY" scripts/query.py value data/results/datasets/malnet/exp_tier1_seed_mi.json::modes::combined::J_maximin_bits --digits 2
run "$PY" scripts/query.py value data/results/datasets/cic_andmal/exp_tier1_seed_mi.json::modes::combined::J_maximin_bits --digits 2
run "$PY" scripts/query.py value data/results/datasets/proxy_v3/exp_tier1_seed_mi.json::modes::opcode::J_maximin_bits --digits 2
run "$PY" scripts/query.py value data/results/datasets/malnet/exp_tier1_seed_mi.json::modes::opcode::J_maximin_bits --digits 2
run "$PY" scripts/query.py value data/results/datasets/cic_andmal/exp_tier1_seed_mi.json::modes::opcode::J_maximin_bits --digits 2
run "$PY" scripts/query.py value data/results/datasets/proxy_v3/exp_tier1_seed_mi.json::modes::fuzzy::J_maximin_bits --digits 2
run "$PY" scripts/query.py value data/results/datasets/malnet/exp_tier1_seed_mi.json::modes::fuzzy::J_maximin_bits --digits 2
run "$PY" scripts/query.py value data/results/datasets/cic_andmal/exp_tier1_seed_mi.json::modes::fuzzy::J_maximin_bits --digits 2
run "$PY" scripts/query.py stat max data/results/datasets/proxy_v3/exp_tier1_seed_mi.json::modes::api::J_maximin_bits data/results/datasets/malnet/exp_tier1_seed_mi.json::modes::api::J_maximin_bits data/results/datasets/cic_andmal/exp_tier1_seed_mi.json::modes::api::J_maximin_bits --digits 2
run "$PY" scripts/query.py value data/results/datasets/proxy_v3/exp_tier1_seed_mi.json::modes::labeled::identity::I_signal_bits --digits 2
run "$PY" scripts/query.py value data/results/datasets/malnet/exp_tier1_seed_mi.json::modes::labeled::identity::I_signal_bits --digits 2
run "$PY" scripts/query.py value data/results/datasets/cic_andmal/exp_tier1_seed_mi.json::modes::labeled::identity::I_signal_bits --digits 2
run "$PY" scripts/query.py stat max-abs 'data/results/datasets/piggyback/exp_tier1_seed_mi.json::modes::*::J_maximin_bits' --digits 2
run "$PY" scripts/query.py value data/results/datasets/piggyback/exp_tier1_seed_mi.json::apps

# extended version: Results, collision and identifiability
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.ident.median_signed_error_max regenerated/numbers.json::malnet.ident.median_signed_error_max regenerated/numbers.json::piggyback.ident.median_signed_error_max regenerated/numbers.json::cic.ident.median_signed_error_max --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.ident.rho::n --sep
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.ident.rho::n --sep
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.ident.rho::n --sep
run "$PY" scripts/query.py value regenerated/numbers.json::cic.ident.rho::n --sep
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.ident.max_abs_error regenerated/numbers.json::malnet.ident.max_abs_error regenerated/numbers.json::piggyback.ident.max_abs_error regenerated/numbers.json::cic.ident.max_abs_error --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.ident.max_abs_error regenerated/numbers.json::malnet.ident.max_abs_error regenerated/numbers.json::piggyback.ident.max_abs_error regenerated/numbers.json::cic.ident.max_abs_error --digits 2
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.ident.rho::rho_median regenerated/numbers.json::malnet.ident.rho::rho_median regenerated/numbers.json::piggyback.ident.rho::rho_median regenerated/numbers.json::cic.ident.rho::rho_median --digits 3
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.ident.rho::rho_median regenerated/numbers.json::malnet.ident.rho::rho_median regenerated/numbers.json::piggyback.ident.rho::rho_median regenerated/numbers.json::cic.ident.rho::rho_median --digits 3
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.ident.rho::rho_p95 regenerated/numbers.json::malnet.ident.rho::rho_p95 regenerated/numbers.json::piggyback.ident.rho::rho_p95 regenerated/numbers.json::cic.ident.rho::rho_p95 --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.ident.rho::rho_p95 regenerated/numbers.json::malnet.ident.rho::rho_p95 regenerated/numbers.json::piggyback.ident.rho::rho_p95 regenerated/numbers.json::cic.ident.rho::rho_p95 --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.ident.rho::max_dstar_spread_factor --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::malnet.ident.rho::max_dstar_spread_factor regenerated/numbers.json::piggyback.ident.rho::max_dstar_spread_factor regenerated/numbers.json::cic.ident.rho::max_dstar_spread_factor --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::malnet.ident.rho::max_dstar_spread_factor regenerated/numbers.json::piggyback.ident.rho::max_dstar_spread_factor regenerated/numbers.json::cic.ident.rho::max_dstar_spread_factor --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.ident.colouring_M::fuzzy::ratio_to_labeled regenerated/numbers.json::malnet.ident.colouring_M::fuzzy::ratio_to_labeled regenerated/numbers.json::piggyback.ident.colouring_M::fuzzy::ratio_to_labeled regenerated/numbers.json::cic.ident.colouring_M::fuzzy::ratio_to_labeled --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.ident.colouring_M::fuzzy::ratio_to_labeled regenerated/numbers.json::malnet.ident.colouring_M::fuzzy::ratio_to_labeled regenerated/numbers.json::piggyback.ident.colouring_M::fuzzy::ratio_to_labeled regenerated/numbers.json::cic.ident.colouring_M::fuzzy::ratio_to_labeled --digits 2
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.ident.colouring_M::api::ratio_to_labeled regenerated/numbers.json::malnet.ident.colouring_M::api::ratio_to_labeled regenerated/numbers.json::piggyback.ident.colouring_M::api::ratio_to_labeled regenerated/numbers.json::cic.ident.colouring_M::api::ratio_to_labeled --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.ident.colouring_M::api::ratio_to_labeled regenerated/numbers.json::malnet.ident.colouring_M::api::ratio_to_labeled regenerated/numbers.json::piggyback.ident.colouring_M::api::ratio_to_labeled regenerated/numbers.json::cic.ident.colouring_M::api::ratio_to_labeled --digits 2
run "$PY" scripts/query.py ratio-stat min regenerated/numbers.json::proxy.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::proxy.ident.colouring_M::fuzzy::d_star_90_median regenerated/numbers.json::malnet.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::malnet.ident.colouring_M::fuzzy::d_star_90_median regenerated/numbers.json::piggyback.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::piggyback.ident.colouring_M::fuzzy::d_star_90_median regenerated/numbers.json::cic.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::cic.ident.colouring_M::fuzzy::d_star_90_median --digits 0
run "$PY" scripts/query.py ratio-stat max regenerated/numbers.json::proxy.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::proxy.ident.colouring_M::fuzzy::d_star_90_median regenerated/numbers.json::malnet.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::malnet.ident.colouring_M::fuzzy::d_star_90_median regenerated/numbers.json::piggyback.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::piggyback.ident.colouring_M::fuzzy::d_star_90_median regenerated/numbers.json::cic.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::cic.ident.colouring_M::fuzzy::d_star_90_median --digits 0
run "$PY" scripts/query.py ratio-stat min regenerated/numbers.json::proxy.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::proxy.ident.colouring_M::api::d_star_90_median regenerated/numbers.json::malnet.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::malnet.ident.colouring_M::api::d_star_90_median regenerated/numbers.json::piggyback.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::piggyback.ident.colouring_M::api::d_star_90_median regenerated/numbers.json::cic.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::cic.ident.colouring_M::api::d_star_90_median --digits 0
run "$PY" scripts/query.py ratio-stat max regenerated/numbers.json::proxy.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::proxy.ident.colouring_M::api::d_star_90_median regenerated/numbers.json::malnet.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::malnet.ident.colouring_M::api::d_star_90_median regenerated/numbers.json::piggyback.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::piggyback.ident.colouring_M::api::d_star_90_median regenerated/numbers.json::cic.ident.colouring_M::labeled::d_star_90_median regenerated/numbers.json::cic.ident.colouring_M::api::d_star_90_median --digits 0
run "$PY" scripts/query.py stat min 'data/results/datasets/malnet_payload/tier4/v3/sdk_transplant.json::median_sdk_token_frac::*' --percent --digits 0
run "$PY" scripts/query.py stat max 'data/results/datasets/malnet_payload/tier4/v3/sdk_transplant.json::median_sdk_token_frac::*' --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.transplant::A2a::sdk --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.transplant::A2a::full --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.transplant::A2a::host --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.transplant::A2a_sdk_vs_host_p --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.transplant::A2c::unseen::follows_sdk --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.transplant::A2c::seen::follows_sdk --percent --digits 0
run "$PY" scripts/query.py stat min 'data/results/datasets/proxy_v3_gneardup/tier4/v3/sdk_transplant.json::median_sdk_token_frac::*' --percent --digits 2
run "$PY" scripts/query.py stat max 'data/results/datasets/proxy_v3_gneardup/tier4/v3/sdk_transplant.json::median_sdk_token_frac::*' --percent --digits 0
run "$PY" scripts/query.py nth-largest 3 'data/results/datasets/proxy_v3_gneardup/tier4/v3/sdk_transplant.json::median_sdk_token_frac::*' --percent --digits 0 --rounding ceiling
run "$PY" scripts/query.py stat count 'data/results/datasets/proxy_v3_gneardup/tier4/v3/sdk_transplant.json::median_sdk_token_frac::*'
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.transplant::A2a::sdk --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.transplant::A2a::full --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.transplant::A2a::host --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.transplant::A2a_sdk_vs_host_p --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.transplant::A2c::unseen::follows_host --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.transplant::A2c::seen::follows_host --percent --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.transplant::A2c::unseen::follows_sdk --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.transplant::A2b::0.3::sdk --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.transplant::A2b::0.7::sdk --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::malnet.transplant::A2b::*::sdk' --digits 2
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::malnet.transplant::A2b::*::sdk' --digits 2
run cat regenerated/tables_extended/rq2_inflation.tex
run "$PY" scripts/query.py stat max 'data/results/datasets/malnet/exp_tier1_inflation.json::k_sweep::*' --sep
run "$PY" scripts/query.py value data/results/datasets/malnet/exp_tier1_inflation.json::dim --sep
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.infl::margin_ratio regenerated/numbers.json::malnet.infl::margin_ratio regenerated/numbers.json::piggyback.infl::margin_ratio --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.infl::margin_ratio regenerated/numbers.json::malnet.infl::margin_ratio regenerated/numbers.json::piggyback.infl::margin_ratio --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.infl::still_correct regenerated/numbers.json::malnet.infl::still_correct regenerated/numbers.json::piggyback.infl::still_correct --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.infl::still_correct regenerated/numbers.json::malnet.infl::still_correct regenerated/numbers.json::piggyback.infl::still_correct --percent --digits 0
run "$PY" scripts/query.py diff-stat max regenerated/numbers.json::proxy.infl::iso0::1048576 regenerated/numbers.json::proxy.infl::isok::1048576 regenerated/numbers.json::malnet.infl::iso0::1048576 regenerated/numbers.json::malnet.infl::isok::1048576 regenerated/numbers.json::piggyback.infl::iso0::1048576 regenerated/numbers.json::piggyback.infl::isok::1048576 --percent --digits 1

# extended version: Results, explainer comparison
run cat regenerated/tables_extended/rq2_localization.tex
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/e2a_standardized.json::labeled::dim --sep
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.loc1024.ratio_by_explainer::lin_exact --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.loc1024.ratio_by_explainer::lin_exact --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.loc1024.ratio_by_explainer::lin_exact --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.loc1024.ratio_by_explainer::*' 'regenerated/numbers.json::malnet.loc1024.ratio_by_explainer::*' 'regenerated/numbers.json::piggyback.loc1024.ratio_by_explainer::*' --skip-key lin_exact --digits 2
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.loc1024.ratio_by_explainer::*' 'regenerated/numbers.json::malnet.loc1024.ratio_by_explainer::*' 'regenerated/numbers.json::piggyback.loc1024.ratio_by_explainer::*' --skip-key lin_exact --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.highdim.ratio_by_dim::1048576 --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.highdim.ratio_by_dim::1048576 --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.highdim.ratio_by_dim::1048576 --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.highdim.p_by_family_maxdim::luminati --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.loc1024.base_by_family::luminati --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::malnet.auroc_by_family::*' regenerated/numbers.json::proxy.auroc_by_family::cashraven regenerated/numbers.json::proxy.auroc_by_family::ipninja regenerated/numbers.json::proxy.auroc_by_family::popa regenerated/numbers.json::proxy.auroc_by_family::traffmonetizer --digits 2
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::malnet.auroc_by_family::*' regenerated/numbers.json::proxy.auroc_by_family::cashraven regenerated/numbers.json::proxy.auroc_by_family::ipninja regenerated/numbers.json::proxy.auroc_by_family::popa regenerated/numbers.json::proxy.auroc_by_family::traffmonetizer --digits 2
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.auroc_by_family::luminati regenerated/numbers.json::proxy.auroc_by_family::ipidea regenerated/numbers.json::proxy.auroc_by_family::monkeysocks regenerated/numbers.json::piggyback.auroc_by_family::piggybacked --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.auroc_by_family::luminati regenerated/numbers.json::proxy.auroc_by_family::ipidea regenerated/numbers.json::proxy.auroc_by_family::monkeysocks regenerated/numbers.json::piggyback.auroc_by_family::piggybacked --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.mx::lin_exact::M3_flip_0.01 --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.mx::lin_exact::M3_flip_0.01 --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.mx::lin_exact::M3_flip_0.01 --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.mx::lin_exact::M3_flip_0.05 regenerated/numbers.json::malnet.mx::lin_exact::M3_flip_0.05 regenerated/numbers.json::piggyback.mx::lin_exact::M3_flip_0.05 --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.mx::lin_exact::M3_flip_0.05 regenerated/numbers.json::malnet.mx::lin_exact::M3_flip_0.05 regenerated/numbers.json::piggyback.mx::lin_exact::M3_flip_0.05 --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.mx::lin_exact::M3_retention_random_0.01 regenerated/numbers.json::malnet.mx::lin_exact::M3_retention_random_0.01 regenerated/numbers.json::piggyback.mx::lin_exact::M3_retention_random_0.01 --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.mx::lin_exact::M3_retention_random_0.01 regenerated/numbers.json::malnet.mx::lin_exact::M3_retention_random_0.01 regenerated/numbers.json::piggyback.mx::lin_exact::M3_retention_random_0.01 --digits 2
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.mx::mlp_ig::M3_flip_0.01 regenerated/numbers.json::malnet.mx::mlp_ig::M3_flip_0.01 regenerated/numbers.json::piggyback.mx::mlp_ig::M3_flip_0.01 regenerated/numbers.json::proxy.mx::mlp_gxi::M3_flip_0.01 regenerated/numbers.json::malnet.mx::mlp_gxi::M3_flip_0.01 regenerated/numbers.json::piggyback.mx::mlp_gxi::M3_flip_0.01 --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.mx::mlp_ig::M3_flip_0.01 regenerated/numbers.json::malnet.mx::mlp_ig::M3_flip_0.01 regenerated/numbers.json::piggyback.mx::mlp_ig::M3_flip_0.01 regenerated/numbers.json::proxy.mx::mlp_gxi::M3_flip_0.01 regenerated/numbers.json::malnet.mx::mlp_gxi::M3_flip_0.01 regenerated/numbers.json::piggyback.mx::mlp_gxi::M3_flip_0.01 --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.mx::rf_path::M3_flip_0.01 --percent --digits 0
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/e2a_standardized.json::labeled::dim --sep
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.mx::*::M5_spearman' 'regenerated/numbers.json::malnet.mx::*::M5_spearman' 'regenerated/numbers.json::piggyback.mx::*::M5_spearman' --digits 3
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.mx::*::M5_spearman' 'regenerated/numbers.json::malnet.mx::*::M5_spearman' 'regenerated/numbers.json::piggyback.mx::*::M5_spearman' --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.mx::lin_exact::M4_top100_jaccard --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.mx::lin_exact::M4_top100_jaccard --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.mx::lin_exact::M4_top100_jaccard --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.mx::rf_path::M4_top100_jaccard --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.seed_spearman::median_spearman_across_seeds --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.seed_spearman::median_spearman_across_seeds --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.seed_spearman::median_spearman_across_seeds --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.seed_spearman::p10 --digits 2
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/p3/renorm.json::n_perm
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/p3/renorm.json::n_apps
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.renorm::spearman regenerated/numbers.json::malnet.renorm::spearman regenerated/numbers.json::piggyback.renorm::spearman --digits 3
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.renorm::spearman regenerated/numbers.json::malnet.renorm::spearman regenerated/numbers.json::piggyback.renorm::spearman --digits 3
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.renorm::rel_l2 regenerated/numbers.json::malnet.renorm::rel_l2 regenerated/numbers.json::piggyback.renorm::rel_l2 --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.renorm::rel_l2 regenerated/numbers.json::malnet.renorm::rel_l2 regenerated/numbers.json::piggyback.renorm::rel_l2 --digits 2
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.renorm::rel_l2_mc_noise regenerated/numbers.json::malnet.renorm::rel_l2_mc_noise regenerated/numbers.json::piggyback.renorm::rel_l2_mc_noise --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.renorm::rel_l2_mc_noise regenerated/numbers.json::malnet.renorm::rel_l2_mc_noise regenerated/numbers.json::piggyback.renorm::rel_l2_mc_noise --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.gnn::macro_f1 --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.gnn::ratio --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.gnn::ratio --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.gnn::macro_f1 --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.gnn::ratio --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.gnnexplainer::exact_ratio --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.gnnexplainer::gnnexplainer_ratio --digits 2
run "$PY" scripts/query.py value data/results/datasets/proxy_v3/exp_xai_gnnexplainer.json::paired_tests::exact_vs_gnnexplainer::wins
run "$PY" scripts/query.py stat sum data/results/datasets/proxy_v3/exp_xai_gnnexplainer.json::paired_tests::exact_vs_gnnexplainer::wins data/results/datasets/proxy_v3/exp_xai_gnnexplainer.json::paired_tests::exact_vs_gnnexplainer::losses
run "$PY" scripts/query.py value data/results/datasets/proxy_v3/exp_xai_gnnexplainer.json::paired_tests::exact_vs_gnnexplainer::p --significant 1
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.gnnexplainer::exact_ratio --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.gnnexplainer::gnnexplainer_ratio --digits 2
run "$PY" scripts/query.py value data/results/datasets/malnet/exp_xai_gnnexplainer.json::paired_tests::exact_vs_gnnexplainer::wins
run "$PY" scripts/query.py stat sum data/results/datasets/malnet/exp_xai_gnnexplainer.json::paired_tests::exact_vs_gnnexplainer::wins data/results/datasets/malnet/exp_xai_gnnexplainer.json::paired_tests::exact_vs_gnnexplainer::losses
run "$PY" scripts/query.py value data/results/datasets/malnet/exp_xai_gnnexplainer.json::paired_tests::exact_vs_gnnexplainer::p --significant 1
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.gnnexplainer::exact_ratio --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.gnnexplainer::gnnexplainer_ratio --digits 2
run "$PY" scripts/query.py value data/results/datasets/piggyback/exp_xai_gnnexplainer.json::paired_tests::exact_vs_gnnexplainer::wins
run "$PY" scripts/query.py stat sum data/results/datasets/piggyback/exp_xai_gnnexplainer.json::paired_tests::exact_vs_gnnexplainer::wins data/results/datasets/piggyback/exp_xai_gnnexplainer.json::paired_tests::exact_vs_gnnexplainer::losses
run "$PY" scripts/query.py value data/results/datasets/piggyback/exp_xai_gnnexplainer.json::paired_tests::exact_vs_gnnexplainer::p --significant 1

# extended version: Results across initial colorings
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.colour::fuzzy|clean|1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.colour::fuzzy|clean|1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.colour::fuzzy|clean|1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.colour::labeled|clean|1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.colour::labeled|clean|1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.colour::labeled|clean|1048576::ratio' --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.colour::api|clean|*::ratio' 'regenerated/numbers.json::malnet.colour::api|clean|*::ratio' 'regenerated/numbers.json::piggyback.colour::api|clean|*::ratio' 'regenerated/numbers.json::proxy.colour::structure|clean|*::ratio' 'regenerated/numbers.json::malnet.colour::structure|clean|*::ratio' 'regenerated/numbers.json::piggyback.colour::structure|clean|*::ratio' --digits 2
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.colour::api|clean|*::ratio' 'regenerated/numbers.json::malnet.colour::api|clean|*::ratio' 'regenerated/numbers.json::piggyback.colour::api|clean|*::ratio' 'regenerated/numbers.json::proxy.colour::structure|clean|*::ratio' 'regenerated/numbers.json::malnet.colour::structure|clean|*::ratio' 'regenerated/numbers.json::piggyback.colour::structure|clean|*::ratio' --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.colour::labeled|renamed|1048576::ratio' 'regenerated/numbers.json::malnet.colour::labeled|renamed|1048576::ratio' 'regenerated/numbers.json::piggyback.colour::labeled|renamed|1048576::ratio' --digits 2
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.colour::labeled|renamed|1048576::ratio' 'regenerated/numbers.json::malnet.colour::labeled|renamed|1048576::ratio' 'regenerated/numbers.json::piggyback.colour::labeled|renamed|1048576::ratio' --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.granularity::fuzzy::*' 'regenerated/numbers.json::malnet.granularity::fuzzy::*' 'regenerated/numbers.json::piggyback.granularity::fuzzy::*' --percent --digits 0
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.granularity::fuzzy::*' 'regenerated/numbers.json::malnet.granularity::fuzzy::*' 'regenerated/numbers.json::piggyback.granularity::fuzzy::*' --percent --digits 0
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.granularity::api::*' 'regenerated/numbers.json::malnet.granularity::api::*' 'regenerated/numbers.json::piggyback.granularity::api::*' --percent --digits 0
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.granularity::api::*' 'regenerated/numbers.json::malnet.granularity::api::*' 'regenerated/numbers.json::piggyback.granularity::api::*' --percent --digits 0
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.granularity::structure::*' 'regenerated/numbers.json::malnet.granularity::structure::*' 'regenerated/numbers.json::piggyback.granularity::structure::*' --percent --digits 0
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.granularity::structure::*' 'regenerated/numbers.json::malnet.granularity::structure::*' 'regenerated/numbers.json::piggyback.granularity::structure::*' --percent --digits 0

# extended version: Results, certificates and structural perturbation
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.renaming_soundness::n_evals --sep
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.renaming_soundness::n_evals --sep
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.renaming_soundness::n_evals --sep
run "$PY" scripts/query.py value regenerated/numbers.json::cic.renaming_soundness::n_evals --sep
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.renaming_soundness::flip_rate regenerated/numbers.json::malnet.renaming_soundness::flip_rate regenerated/numbers.json::piggyback.renaming_soundness::flip_rate regenerated/numbers.json::cic.renaming_soundness::flip_rate --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.renaming_soundness::flip_rate regenerated/numbers.json::malnet.renaming_soundness::flip_rate regenerated/numbers.json::piggyback.renaming_soundness::flip_rate regenerated/numbers.json::cic.renaming_soundness::flip_rate --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.renaming_soundness::delta_over_radius_median --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::malnet.renaming_soundness::delta_over_radius_median regenerated/numbers.json::piggyback.renaming_soundness::delta_over_radius_median regenerated/numbers.json::cic.renaming_soundness::delta_over_radius_median --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::malnet.renaming_soundness::delta_over_radius_median regenerated/numbers.json::piggyback.renaming_soundness::delta_over_radius_median regenerated/numbers.json::cic.renaming_soundness::delta_over_radius_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.renaming_soundness::radius_median --digits 3
run "$PY" scripts/query.py stat min regenerated/numbers.json::malnet.renaming_soundness::radius_median regenerated/numbers.json::piggyback.renaming_soundness::radius_median regenerated/numbers.json::cic.renaming_soundness::radius_median --digits 3
run "$PY" scripts/query.py stat max regenerated/numbers.json::malnet.renaming_soundness::radius_median regenerated/numbers.json::piggyback.renaming_soundness::radius_median regenerated/numbers.json::cic.renaming_soundness::radius_median --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.thm_bounds::apps
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.thm_bounds::apps
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.thm_bounds::apps
run "$PY" scripts/query.py value regenerated/numbers.json::cic.thm_bounds::apps
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.thm_bounds::summary::labeled|*::log10_slack_global_median' --digits 1
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.thm_bounds::summary::labeled|*::log10_slack_global_median' --digits 1
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::malnet.thm_bounds::summary::labeled|*::log10_slack_global_median' --digits 1
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::malnet.thm_bounds::summary::labeled|*::log10_slack_global_median' --digits 1
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::piggyback.thm_bounds::summary::labeled|*::log10_slack_global_median' 'regenerated/numbers.json::cic.thm_bounds::summary::labeled|*::log10_slack_global_median' --digits 1
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::piggyback.thm_bounds::summary::labeled|*::log10_slack_global_median' 'regenerated/numbers.json::cic.thm_bounds::summary::labeled|*::log10_slack_global_median' --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.thm_bounds::dmax --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.thm_bounds::dmax --digits 0 --sep
run "$PY" scripts/query.py stat min regenerated/numbers.json::piggyback.thm_bounds::dmax regenerated/numbers.json::cic.thm_bounds::dmax --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::piggyback.thm_bounds::dmax regenerated/numbers.json::cic.thm_bounds::dmax --digits 0
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.thm_bounds::summary::*::L_le_bound_global_frac' 'regenerated/numbers.json::malnet.thm_bounds::summary::*::L_le_bound_global_frac' 'regenerated/numbers.json::piggyback.thm_bounds::summary::*::L_le_bound_global_frac' 'regenerated/numbers.json::cic.thm_bounds::summary::*::L_le_bound_global_frac'
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.r4::flip_labeled --percent --digits 1
run "$PY" scripts/query.py stat min regenerated/numbers.json::malnet.r4::flip_labeled regenerated/numbers.json::piggyback.r4::flip_labeled regenerated/numbers.json::cic.r4::flip_labeled --percent --digits 1
run "$PY" scripts/query.py stat max regenerated/numbers.json::malnet.r4::flip_labeled regenerated/numbers.json::piggyback.r4::flip_labeled regenerated/numbers.json::cic.r4::flip_labeled --percent --digits 1
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.r4::flip_*' 'regenerated/numbers.json::malnet.r4::flip_*' 'regenerated/numbers.json::piggyback.r4::flip_*' 'regenerated/numbers.json::cic.r4::flip_*' --percent --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.r4::ret_labeled_sdk --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.r4::ret_labeled_host --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.r4::ret_labeled_sdk --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.r4::ret_labeled_host --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.r4::flatten_*_all' 'regenerated/numbers.json::malnet.r4::flatten_*_all' --percent --digits 0
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.r4::flatten_*_all' 'regenerated/numbers.json::malnet.r4::flatten_*_all' --percent --digits 0
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::cic.r4::flatten_*_all' --percent --digits 0
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::cic.r4::flatten_*_all' --percent --digits 0
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::piggyback.r4::flatten_*_all' --percent --digits 0
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::piggyback.r4::flatten_*_all' --percent --digits 0
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.r4::flatten_*_host' 'regenerated/numbers.json::malnet.r4::flatten_*_host' 'regenerated/numbers.json::piggyback.r4::flatten_*_host' 'regenerated/numbers.json::cic.r4::flatten_*_host' --percent --digits 0
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.r4::flatten_*_host' 'regenerated/numbers.json::malnet.r4::flatten_*_host' 'regenerated/numbers.json::piggyback.r4::flatten_*_host' 'regenerated/numbers.json::cic.r4::flatten_*_host' --percent --digits 0
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.r4::flatten_*_sdk' 'regenerated/numbers.json::cic.r4::flatten_*_sdk' --percent --digits 0
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::malnet.r4::flatten_*_sdk' --percent --digits 0
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::piggyback.r4::flatten_*_sdk' --percent --digits 0

# extended version: Results, the accuracy-robustness frontier
run cat regenerated/tables_extended/rq3_frontier.tex
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/e2a_standardized.json::labeled::dim --sep
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.tightness::labeled::k_grad_median --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.tightness::labeled::k_grad_median --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.tightness::labeled::k_grad_median --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::cic.tightness::labeled::k_grad_median --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.tightness::labeled::k_steer_median --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.tightness::labeled::k_steer_median --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.tightness::labeled::k_steer_median --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::cic.tightness::labeled::k_steer_median --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.tightness::labeled::k_cert_median --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.tightness::labeled::k_cert_median --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.tightness::labeled::k_cert_median --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::cic.tightness::labeled::k_cert_median --digits 0 --sep
run "$PY" scripts/query.py above-one min regenerated/numbers.json::proxy.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::malnet.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::piggyback.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::cic.tightness::labeled::ceiling_over_floor_median --percent --digits 0
run "$PY" scripts/query.py above-one max regenerated/numbers.json::proxy.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::malnet.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::piggyback.tightness::labeled::ceiling_over_floor_median regenerated/numbers.json::cic.tightness::labeled::ceiling_over_floor_median --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.tightness::labeled::ceiling_over_floor_median --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.tightness::labeled::ceiling_over_floor_median --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.tightness::labeled::ceiling_over_floor_median --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::cic.tightness::labeled::ceiling_over_floor_median --digits 2
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/v3/cert_tightness.json::C14::n_salts
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.salt_spearman regenerated/numbers.json::malnet.salt_spearman regenerated/numbers.json::piggyback.salt_spearman regenerated/numbers.json::cic.salt_spearman --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.salt_spearman regenerated/numbers.json::malnet.salt_spearman regenerated/numbers.json::piggyback.salt_spearman regenerated/numbers.json::cic.salt_spearman --digits 2
run "$PY" scripts/query.py ratio-stat min regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::malnet.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::piggyback.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::cic.tightness::labeled::k_grad_median --digits 0
run "$PY" scripts/query.py ratio-stat max regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::malnet.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::piggyback.tightness::labeled::k_grad_median regenerated/numbers.json::proxy.tightness::labeled::k_grad_median regenerated/numbers.json::cic.tightness::labeled::k_grad_median --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.frontier.fuzzy::k_cert_median regenerated/numbers.json::proxy.frontier.api::k_cert_median regenerated/numbers.json::proxy.frontier.degree::k_cert_median regenerated/numbers.json::malnet.frontier.fuzzy::k_cert_median regenerated/numbers.json::malnet.frontier.api::k_cert_median regenerated/numbers.json::malnet.frontier.degree::k_cert_median --digits 0
run "$PY" scripts/query.py above-one min regenerated/numbers.json::piggyback.tightness::fuzzy::ceiling_over_floor_median regenerated/numbers.json::piggyback.tightness::api::ceiling_over_floor_median regenerated/numbers.json::piggyback.tightness::structure::ceiling_over_floor_median regenerated/numbers.json::cic.tightness::fuzzy::ceiling_over_floor_median regenerated/numbers.json::cic.tightness::api::ceiling_over_floor_median regenerated/numbers.json::cic.tightness::structure::ceiling_over_floor_median --percent --digits 0
run "$PY" scripts/query.py above-one max regenerated/numbers.json::piggyback.tightness::fuzzy::ceiling_over_floor_median regenerated/numbers.json::piggyback.tightness::api::ceiling_over_floor_median regenerated/numbers.json::piggyback.tightness::structure::ceiling_over_floor_median regenerated/numbers.json::cic.tightness::fuzzy::ceiling_over_floor_median regenerated/numbers.json::cic.tightness::api::ceiling_over_floor_median regenerated/numbers.json::cic.tightness::structure::ceiling_over_floor_median --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.frontier.identifier::rand_never_frac --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.frontier.identifier::rand_never_frac --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.frontier.identifier::rand_never_frac --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.frontier.identifier::rand_never_frac --percent --digits 0
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/v3/attack_realism.json::budget
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.realism::labeled::white_box --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.realism::labeled::white_box --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::piggyback.realism::labeled::white_box regenerated/numbers.json::cic.realism::labeled::white_box --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::piggyback.realism::labeled::white_box regenerated/numbers.json::cic.realism::labeled::white_box --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.realism::labeled::greybox_disjoint --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.realism::labeled::greybox_disjoint --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.realism::labeled::greybox_disjoint --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.realism::labeled::greybox_disjoint --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.realism::labeled::transfer_reseed regenerated/numbers.json::malnet.realism::labeled::transfer_reseed regenerated/numbers.json::piggyback.realism::labeled::transfer_reseed regenerated/numbers.json::cic.realism::labeled::transfer_reseed --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.realism::labeled::transfer_reseed regenerated/numbers.json::malnet.realism::labeled::transfer_reseed regenerated/numbers.json::piggyback.realism::labeled::transfer_reseed regenerated/numbers.json::cic.realism::labeled::transfer_reseed --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.realism::labeled::random --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.realism::labeled::random --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.realism::labeled::random --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.realism::labeled::random --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.realism::structure::white_box --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::malnet.realism::structure::white_box regenerated/numbers.json::piggyback.realism::structure::white_box regenerated/numbers.json::cic.realism::structure::white_box --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::malnet.realism::structure::white_box regenerated/numbers.json::piggyback.realism::structure::white_box regenerated/numbers.json::cic.realism::structure::white_box --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.steer_graph::never_flipped_frac --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.steer_graph::never_flipped_frac --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.steer_graph::never_flipped_frac --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.steer_graph::never_flipped_frac --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.steer_graph::k_graph_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.steer_graph::k_graph_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.steer_graph::k_graph_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.steer_graph::k_graph_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.frontier.combined::k_steer_median --digits 0 --sep
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.frontier.fuzzy::k_steer_median regenerated/numbers.json::proxy.frontier.api::k_steer_median regenerated/numbers.json::proxy.frontier.degree::k_steer_median --digits 0 --sep
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.frontier.fuzzy::k_steer_median regenerated/numbers.json::proxy.frontier.api::k_steer_median regenerated/numbers.json::proxy.frontier.degree::k_steer_median --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.frontier.identifier::k_steer_median --digits 0 --sep
run "$PY" scripts/query.py diff-stat min regenerated/numbers.json::proxy.frontier.identifier::f1 regenerated/numbers.json::proxy.frontier.combined::f1 regenerated/numbers.json::proxy.frontier.identifier::f1 regenerated/numbers.json::proxy.frontier.fuzzy::f1 regenerated/numbers.json::proxy.frontier.identifier::f1 regenerated/numbers.json::proxy.frontier.api::f1 regenerated/numbers.json::proxy.frontier.identifier::f1 regenerated/numbers.json::proxy.frontier.degree::f1 --digits 2
run "$PY" scripts/query.py diff-stat max regenerated/numbers.json::proxy.frontier.identifier::f1 regenerated/numbers.json::proxy.frontier.combined::f1 regenerated/numbers.json::proxy.frontier.identifier::f1 regenerated/numbers.json::proxy.frontier.fuzzy::f1 regenerated/numbers.json::proxy.frontier.identifier::f1 regenerated/numbers.json::proxy.frontier.api::f1 regenerated/numbers.json::proxy.frontier.identifier::f1 regenerated/numbers.json::proxy.frontier.degree::f1 --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.frontier.combined::k_steer_median --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::malnet.frontier.fuzzy::k_steer_median regenerated/numbers.json::malnet.frontier.api::k_steer_median regenerated/numbers.json::malnet.frontier.degree::k_steer_median --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::malnet.frontier.fuzzy::k_steer_median regenerated/numbers.json::malnet.frontier.api::k_steer_median regenerated/numbers.json::malnet.frontier.degree::k_steer_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.frontier.identifier::k_steer_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.frontier.combined::f1 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.frontier.identifier::f1 --digits 3
run "$PY" scripts/query.py diff-stat max regenerated/numbers.json::piggyback.frontier.identifier::f1 regenerated/numbers.json::piggyback.frontier.combined::f1 regenerated/numbers.json::piggyback.frontier.identifier::f1 regenerated/numbers.json::piggyback.frontier.fuzzy::f1 regenerated/numbers.json::piggyback.frontier.identifier::f1 regenerated/numbers.json::piggyback.frontier.api::f1 regenerated/numbers.json::piggyback.frontier.identifier::f1 regenerated/numbers.json::piggyback.frontier.degree::f1 regenerated/numbers.json::cic.frontier.identifier::f1 regenerated/numbers.json::cic.frontier.combined::f1 regenerated/numbers.json::cic.frontier.identifier::f1 regenerated/numbers.json::cic.frontier.fuzzy::f1 regenerated/numbers.json::cic.frontier.identifier::f1 regenerated/numbers.json::cic.frontier.api::f1 regenerated/numbers.json::cic.frontier.identifier::f1 regenerated/numbers.json::cic.frontier.degree::f1 --digits 2
run "$PY" scripts/query.py ratio-stat min regenerated/numbers.json::piggyback.frontier.combined::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::piggyback.frontier.fuzzy::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::piggyback.frontier.api::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::piggyback.frontier.degree::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.combined::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.fuzzy::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.api::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.degree::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median --digits 1
run "$PY" scripts/query.py ratio-stat max regenerated/numbers.json::piggyback.frontier.combined::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::piggyback.frontier.fuzzy::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::piggyback.frontier.api::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::piggyback.frontier.degree::k_steer_median regenerated/numbers.json::piggyback.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.combined::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.fuzzy::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.api::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median regenerated/numbers.json::cic.frontier.degree::k_steer_median regenerated/numbers.json::cic.frontier.identifier::k_steer_median --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.linf::1.0::k_cert_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.linf::0.25::k_cert_median --digits 0
run "$PY" scripts/query.py diff regenerated/numbers.json::proxy.linf::1.0::f1 regenerated/numbers.json::proxy.linf::0.25::f1 --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.linf::1.0::k_cert_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.linf::0.25::k_cert_median --digits 0
run "$PY" scripts/query.py diff regenerated/numbers.json::malnet.linf::1.0::f1 regenerated/numbers.json::malnet.linf::0.25::f1 --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.linf::1.0::k_cert_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.linf::0.25::k_cert_median --digits 0
run "$PY" scripts/query.py diff regenerated/numbers.json::piggyback.linf::1.0::f1 regenerated/numbers.json::piggyback.linf::0.25::f1 --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::cic.linf::1.0::k_cert_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.linf::0.25::k_cert_median --digits 0
run "$PY" scripts/query.py diff regenerated/numbers.json::cic.linf::1.0::f1 regenerated/numbers.json::cic.linf::0.25::f1 --digits 2
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup_fcg/tier4/e2a_standardized.json::labeled::dim --sep
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.monotone::*::auc_unconstrained' 'regenerated/numbers.json::malnet.monotone::*::auc_unconstrained' 'regenerated/numbers.json::cic.monotone::*::auc_unconstrained' --digits 2
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.monotone::*::auc_unconstrained' 'regenerated/numbers.json::malnet.monotone::*::auc_unconstrained' 'regenerated/numbers.json::cic.monotone::*::auc_unconstrained' --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.monotone::*::auc_monotone' 'regenerated/numbers.json::malnet.monotone::*::auc_monotone' 'regenerated/numbers.json::cic.monotone::*::auc_monotone' --digits 2
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.monotone::*::auc_monotone' 'regenerated/numbers.json::malnet.monotone::*::auc_monotone' 'regenerated/numbers.json::cic.monotone::*::auc_monotone' --digits 2
run "$PY" scripts/query.py diff-stat min 'regenerated/numbers.json::proxy.monotone::*::auc_unconstrained' 'regenerated/numbers.json::proxy.monotone::*::auc_monotone' --percent --digits 0
run "$PY" scripts/query.py diff-stat max 'regenerated/numbers.json::proxy.monotone::*::auc_unconstrained' 'regenerated/numbers.json::proxy.monotone::*::auc_monotone' --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.tree_cert::zero_cert_frac regenerated/numbers.json::malnet.tree_cert::zero_cert_frac regenerated/numbers.json::piggyback.tree_cert::zero_cert_frac regenerated/numbers.json::cic.tree_cert::zero_cert_frac --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.tree_cert::zero_cert_frac regenerated/numbers.json::malnet.tree_cert::zero_cert_frac regenerated/numbers.json::piggyback.tree_cert::zero_cert_frac regenerated/numbers.json::cic.tree_cert::zero_cert_frac --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.tree_cert::k_cert_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.tree_cert::k_cert_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.tree_cert::k_cert_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.tree_cert::k_cert_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.tree_cert::k_attack_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.tree_cert::k_attack_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.tree_cert::k_attack_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.tree_cert::k_attack_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.evenness::e1_spearman --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.evenness::e1_spearman --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::cic.evenness::e1_spearman --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.evenness::e1_spearman --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.evenness::e1_p regenerated/numbers.json::malnet.evenness::e1_p regenerated/numbers.json::piggyback.evenness::e1_p regenerated/numbers.json::cic.evenness::e1_p --sci 1
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.evenness::per_family::*' --digits 2
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.evenness::per_family::*' --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::malnet.evenness::per_family::*' --digits 2
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::malnet.evenness::per_family::*' --digits 2
run "$PY" scripts/query.py value data/results/datasets/malnet/exp_tier3_certificates.json::smoothing::deletion_prob
run "$PY" scripts/query.py value data/results/datasets/malnet/exp_tier3_certificates.json::smoothing::n_samples
run "$PY" scripts/query.py value data/results/datasets/malnet/exp_tier3_certificates.json::smoothing::alpha
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.smoothing::abstain --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.smoothing::abstain --percent --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.smoothing::abstain --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.smoothing::abstain --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.smoothing::passes
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.drebin_cert::linear_svm::f1 regenerated/numbers.json::malnet.drebin_cert::linear_svm::f1 regenerated/numbers.json::piggyback.drebin_cert::linear_svm::f1 regenerated/numbers.json::cic.drebin_cert::linear_svm::f1 --digits 3
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.drebin_cert::linear_svm::f1 regenerated/numbers.json::malnet.drebin_cert::linear_svm::f1 regenerated/numbers.json::piggyback.drebin_cert::linear_svm::f1 regenerated/numbers.json::cic.drebin_cert::linear_svm::f1 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.drebin_cert::linear_svm::k_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.drebin_cert::linear_svm::k_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.drebin_cert::linear_svm::k_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.drebin_cert::linear_svm::k_median --digits 0
run "$PY" scripts/query.py stat sum regenerated/numbers.json::proxy.drebin_cert::linear_svm::unconditional regenerated/numbers.json::malnet.drebin_cert::linear_svm::unconditional regenerated/numbers.json::piggyback.drebin_cert::linear_svm::unconditional regenerated/numbers.json::cic.drebin_cert::linear_svm::unconditional --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.mama_cert::mama_package_std::k_median_when_finite --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.mama_cert::mama_family_std::k_median_when_finite --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.mama_cert::mama_package_std::k_median_when_finite --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.mama_cert::mama_family_std::k_median_when_finite --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.mama_cert::mama_package_std::k_median_when_finite --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.mama_cert::mama_family_std::k_median_when_finite --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.mama_cert::mama_package_std::k_median_when_finite --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.mama_cert::mama_family_std::k_median_when_finite --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.mama_cert::mama_package::k_median_when_finite --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.mama_cert::mama_family::k_median_when_finite --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.mama_cert::mama_package::k_median_when_finite --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.mama_cert::mama_family::k_median_when_finite --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.mama_cert::mama_package::k_median_when_finite --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.mama_cert::mama_family::k_median_when_finite --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.mama_cert::mama_package::k_median_when_finite --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.mama_cert::mama_family::k_median_when_finite --digits 0

# extended version: Results, the open world
run cat regenerated/tables_extended/open_world.tex
run "$PY" scripts/query.py count data/manifests/wild_play.csv --where proxy_marker_found=0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.open.O2::linear::macro_f1 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.open.O2::linear::macro_f1 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.open.O2::linear::macro_f1 --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::cic.open.O2::linear::macro_f1 --digits 3
run "$PY" scripts/query.py count 'regenerated/numbers.json::proxy.open.O2::*' 'regenerated/numbers.json::malnet.open.O2::*' 'regenerated/numbers.json::piggyback.open.O2::*' 'regenerated/numbers.json::cic.open.O2::*' --below p=0.05 --above diff_vs_linear=0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.open.O1::linear::wild_fp_rate regenerated/numbers.json::malnet.open.O1::linear::wild_fp_rate regenerated/numbers.json::piggyback.open.O1::linear::wild_fp_rate regenerated/numbers.json::cic.open.O1::linear::wild_fp_rate --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.open.O1::linear::wild_fp_rate regenerated/numbers.json::malnet.open.O1::linear::wild_fp_rate regenerated/numbers.json::piggyback.open.O1::linear::wild_fp_rate regenerated/numbers.json::cic.open.O1::linear::wild_fp_rate --percent --digits 0
run "$PY" scripts/query.py count regenerated/numbers.json::proxy.open.O2::rf regenerated/numbers.json::malnet.open.O2::rf regenerated/numbers.json::piggyback.open.O2::rf regenerated/numbers.json::cic.open.O2::rf --below p=0.05 --below diff_vs_linear=0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.open.O1::rf::wild_fp_rate regenerated/numbers.json::malnet.open.O1::rf::wild_fp_rate regenerated/numbers.json::cic.open.O1::rf::wild_fp_rate --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.open.O1::rf::wild_fp_rate regenerated/numbers.json::malnet.open.O1::rf::wild_fp_rate regenerated/numbers.json::cic.open.O1::rf::wild_fp_rate --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.open.O1::rf::wild_fp_rate --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.open.O1::linear::wild_fp_rate --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.open.abstain::auroc_proxy_vs_wild regenerated/numbers.json::malnet.open.abstain::auroc_proxy_vs_wild regenerated/numbers.json::cic.open.abstain::auroc_proxy_vs_wild --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.open.abstain::auroc_proxy_vs_wild regenerated/numbers.json::malnet.open.abstain::auroc_proxy_vs_wild regenerated/numbers.json::cic.open.abstain::auroc_proxy_vs_wild --digits 2
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.open.abstain::wild_fp_at_tau regenerated/numbers.json::malnet.open.abstain::wild_fp_at_tau regenerated/numbers.json::cic.open.abstain::wild_fp_at_tau --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.open.abstain::wild_fp_at_tau regenerated/numbers.json::malnet.open.abstain::wild_fp_at_tau regenerated/numbers.json::cic.open.abstain::wild_fp_at_tau --percent --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.openset_dex_msp::auroc --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.openset_dex_msp::fpr_at_tpr95 --percent --digits 1
run "$PY" scripts/query.py count data/manifests/wild_play.csv --where proxy_marker_found=0 --where fcg_vector=1 --sep
run "$PY" scripts/query.py count data/manifests/proxy.csv --where closed_world=0 --sep
run "$PY" scripts/query.py stat min regenerated/numbers.json::malnet.wild_colorings::combined::wild_fp regenerated/numbers.json::piggyback.wild_colorings::combined::wild_fp regenerated/numbers.json::cic.wild_colorings::combined::wild_fp --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::malnet.wild_colorings::combined::wild_fp regenerated/numbers.json::piggyback.wild_colorings::combined::wild_fp regenerated/numbers.json::cic.wild_colorings::combined::wild_fp --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::malnet.wild_colorings::fuzzy::wild_fp regenerated/numbers.json::piggyback.wild_colorings::fuzzy::wild_fp regenerated/numbers.json::cic.wild_colorings::fuzzy::wild_fp --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::malnet.wild_colorings::fuzzy::wild_fp regenerated/numbers.json::piggyback.wild_colorings::fuzzy::wild_fp regenerated/numbers.json::cic.wild_colorings::fuzzy::wild_fp --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::malnet.wild_colorings::labeled::wild_fp regenerated/numbers.json::piggyback.wild_colorings::labeled::wild_fp regenerated/numbers.json::cic.wild_colorings::labeled::wild_fp --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::malnet.wild_colorings::labeled::wild_fp regenerated/numbers.json::piggyback.wild_colorings::labeled::wild_fp regenerated/numbers.json::cic.wild_colorings::labeled::wild_fp --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.wild_colorings::combined::wild_fp regenerated/numbers.json::proxy.wild_colorings::fuzzy::wild_fp regenerated/numbers.json::proxy.wild_colorings::api::wild_fp regenerated/numbers.json::proxy.wild_colorings::structure::wild_fp --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.wild_colorings::combined::wild_fp regenerated/numbers.json::proxy.wild_colorings::fuzzy::wild_fp regenerated/numbers.json::proxy.wild_colorings::api::wild_fp regenerated/numbers.json::proxy.wild_colorings::structure::wild_fp --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.wild_colorings::combined::wild_fp --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.wild_colorings::labeled::wild_fp --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.wild_colorings::combined::abstain::auroc regenerated/numbers.json::proxy.wild_colorings::fuzzy::abstain::auroc regenerated/numbers.json::proxy.wild_colorings::api::abstain::auroc regenerated/numbers.json::proxy.wild_colorings::structure::abstain::auroc --digits 2
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.wild_colorings::combined::abstain::auroc regenerated/numbers.json::proxy.wild_colorings::fuzzy::abstain::auroc regenerated/numbers.json::proxy.wild_colorings::api::abstain::auroc regenerated/numbers.json::proxy.wild_colorings::structure::abstain::auroc --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.open.O3::untargeted_lands_benign --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.open.O3::attack_benign_median --digits 0 --rounding half-even
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.open.O3::attack_any_median --digits 0 --rounding half-even
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.open.O3::untargeted_lands_benign --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.open.O3::untargeted_lands_benign --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.open.O3::attack_any_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.open.O3::attack_any_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.open.O3::attack_benign_median --digits 0 --rounding half-even
run "$PY" scripts/query.py value regenerated/numbers.json::cic.open.O3::attack_benign_median --digits 0 --rounding half-even
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.open.O3::attack_benign_median --digits 0 --rounding half-even
run "$PY" scripts/query.py value data/results/datasets/proxy_v3_gneardup/tier4/v3/bundle_attack.json::n_donors
run "$PY" scripts/query.py stat min data/results/datasets/proxy_v3_gneardup/tier4/v3/bundle_attack.json::n_packages data/results/datasets/malnet/tier4/v3/bundle_attack.json::n_packages data/results/datasets/piggyback/tier4/v3/bundle_attack.json::n_packages data/results/datasets/cic_andmal/tier4/v3/bundle_attack.json::n_packages --sep
run "$PY" scripts/query.py stat max data/results/datasets/proxy_v3_gneardup/tier4/v3/bundle_attack.json::n_packages data/results/datasets/malnet/tier4/v3/bundle_attack.json::n_packages data/results/datasets/piggyback/tier4/v3/bundle_attack.json::n_packages data/results/datasets/cic_andmal/tier4/v3/bundle_attack.json::n_packages --sep
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.bundle::greedy::success --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.bundle::greedy::median_tokens_when_successful --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.bundle::greedy_packages_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.bundle::greedy::success --percent --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::cic.bundle::greedy::success --percent --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.bundle::greedy::median_tokens_when_successful --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::cic.bundle::greedy::median_tokens_when_successful --digits 0 --sep
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.bundle::greedy_packages_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::cic.bundle::greedy_packages_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.bundle::greedy::success --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.bundle::greedy_packages_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.bundle::greedy::median_tokens_when_successful --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.bundle::random::success --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::malnet.bundle::random::success regenerated/numbers.json::piggyback.bundle::random::success regenerated/numbers.json::cic.bundle::random::success --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::malnet.bundle::random::success regenerated/numbers.json::piggyback.bundle::random::success regenerated/numbers.json::cic.bundle::random::success --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.bundle::whole_app_flip_rate_mean --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::malnet.bundle::whole_app_flip_rate_mean regenerated/numbers.json::piggyback.bundle::whole_app_flip_rate_mean regenerated/numbers.json::cic.bundle::whole_app_flip_rate_mean --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::malnet.bundle::whole_app_flip_rate_mean regenerated/numbers.json::piggyback.bundle::whole_app_flip_rate_mean regenerated/numbers.json::cic.bundle::whole_app_flip_rate_mean --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.temporal::gap --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.temporal::gap --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.temporal::gap --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::cic.temporal::gap --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.temporal::f1_temporal --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.temporal::f1_random --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.temporal::f1_temporal --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.temporal::f1_random --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.temporal::f1_temporal --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.temporal::f1_random --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::cic.temporal::f1_temporal --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::cic.temporal::f1_random --digits 3
run "$PY" scripts/query.py weighted-mean 'data/results/datasets/proxy_v3_gneardup_fcg/tier4/v4/temporal_drift.json::fields::first_seen::coverage::*::dated' 'data/results/datasets/proxy_v3_gneardup_fcg/tier4/v4/temporal_drift.json::fields::first_seen::coverage::*::n' --skip-key non_proxy --percent --digits 0
run "$PY" scripts/query.py stat count 'data/results/datasets/proxy_v3_gneardup_fcg/tier4/v4/temporal_drift.json::fields::first_seen::coverage::*::n' --skip-key non_proxy

# extended version: Results, control-flow graphs and the deployed classifier
run cat regenerated/tables_extended/cfg_deployed.tex
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_cert::concat|identifier::f1' --digits 3
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_cert::fcg_block|identifier::f1' --digits 3
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_cert::cfg|identifier::f1' --digits 3
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_cert::concat|identifier::f1' --digits 3
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_cert::fcg_block|identifier::f1' --digits 3
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_cert::cfg|identifier::f1' --digits 3
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_cert::concat|identifier::f1' --digits 3
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_cert::fcg_block|identifier::f1' --digits 3
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_cert::cfg|identifier::f1' --digits 3
run "$PY" scripts/query.py value 'regenerated/numbers.json::cic.cfg_cert::concat|identifier::f1' --digits 3
run "$PY" scripts/query.py value 'regenerated/numbers.json::cic.cfg_cert::fcg_block|identifier::f1' --digits 3
run "$PY" scripts/query.py value 'regenerated/numbers.json::cic.cfg_cert::cfg|identifier::f1' --digits 3
run "$PY" scripts/query.py stat sum 'regenerated/numbers.json::proxy.cfg_cert::*::violations' 'regenerated/numbers.json::malnet.cfg_cert::*::violations' 'regenerated/numbers.json::piggyback.cfg_cert::*::violations' 'regenerated/numbers.json::cic.cfg_cert::*::violations'
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.cfg_cert::cfg|identifier::attack_over_cert_median' 'regenerated/numbers.json::malnet.cfg_cert::cfg|identifier::attack_over_cert_median' 'regenerated/numbers.json::piggyback.cfg_cert::cfg|identifier::attack_over_cert_median' 'regenerated/numbers.json::cic.cfg_cert::cfg|identifier::attack_over_cert_median' --digits 2
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.cfg_cert::cfg|identifier::attack_over_cert_median' 'regenerated/numbers.json::malnet.cfg_cert::cfg|identifier::attack_over_cert_median' 'regenerated/numbers.json::piggyback.cfg_cert::cfg|identifier::attack_over_cert_median' 'regenerated/numbers.json::cic.cfg_cert::cfg|identifier::attack_over_cert_median' --digits 2
run "$PY" scripts/query.py ratio 'regenerated/numbers.json::proxy.cfg_cert::cfg|identifier::k_certgrad_median' 'regenerated/numbers.json::proxy.cfg_cert::fcg_block|identifier::k_certgrad_median' --digits 1
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_cert::cfg|identifier::k_certgrad_median' --digits 0 --rounding half-even --sep
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_cert::fcg_block|identifier::k_certgrad_median' --digits 0 --rounding half-even --sep
run "$PY" scripts/query.py ratio 'regenerated/numbers.json::malnet.cfg_cert::cfg|identifier::k_certgrad_median' 'regenerated/numbers.json::malnet.cfg_cert::fcg_block|identifier::k_certgrad_median' --digits 1
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_cert::cfg|identifier::k_certgrad_median' --digits 0 --rounding half-even --sep
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_cert::fcg_block|identifier::k_certgrad_median' --digits 0 --rounding half-even --sep
run "$PY" scripts/query.py ratio 'regenerated/numbers.json::piggyback.cfg_cert::cfg|identifier::k_certgrad_median' 'regenerated/numbers.json::piggyback.cfg_cert::fcg_block|identifier::k_certgrad_median' --digits 1
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_cert::cfg|identifier::k_certgrad_median' --digits 0 --rounding half-even --sep
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_cert::fcg_block|identifier::k_certgrad_median' --digits 0 --rounding half-even --sep
run "$PY" scripts/query.py ratio 'regenerated/numbers.json::cic.cfg_cert::cfg|identifier::k_certgrad_median' 'regenerated/numbers.json::cic.cfg_cert::fcg_block|identifier::k_certgrad_median' --digits 1
run "$PY" scripts/query.py value 'regenerated/numbers.json::cic.cfg_cert::cfg|identifier::k_certgrad_median' --digits 0 --rounding half-even --sep
run "$PY" scripts/query.py value 'regenerated/numbers.json::cic.cfg_cert::fcg_block|identifier::k_certgrad_median' --digits 0 --rounding half-even --sep
run "$PY" scripts/query.py value 'regenerated/numbers.json::cic.cfg_cert::concat|identifier::k_block_median' --digits 0 --rounding half-even
run "$PY" scripts/query.py value 'regenerated/numbers.json::cic.cfg_cert::concat|identifier::k_attack_median' --digits 0 --rounding half-even
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_cert::concat|identifier::k_block_median' --digits 0 --rounding half-even
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_cert::concat|identifier::k_attack_median' --digits 0 --rounding half-even
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_cert::concat|identifier::k_block_median' --digits 0 --rounding half-even
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_cert::concat|identifier::k_attack_median' --digits 0 --rounding half-even
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_cert::concat|identifier::k_block_median' --digits 0 --rounding half-even
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_cert::concat|identifier::k_attack_median' --digits 0 --rounding half-even
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_steer::cfg|b16::flipped'
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_steer::cfg|b16::n'
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_steer::cfg|b16::flipped'
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_steer::cfg|b16::n'
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_steer::cfg|b16::flipped'
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_steer::cfg|b16::n'
run "$PY" scripts/query.py value 'regenerated/numbers.json::cic.cfg_steer::cfg|b16::flipped'
run "$PY" scripts/query.py value 'regenerated/numbers.json::cic.cfg_steer::cfg|b16::n'
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.cfg_steer::cfg|b16::k_median_flipped' 'regenerated/numbers.json::malnet.cfg_steer::cfg|b16::k_median_flipped' 'regenerated/numbers.json::piggyback.cfg_steer::cfg|b16::k_median_flipped' 'regenerated/numbers.json::cic.cfg_steer::cfg|b16::k_median_flipped' --digits 0
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.cfg_steer::cfg|b16::k_median_flipped' 'regenerated/numbers.json::malnet.cfg_steer::cfg|b16::k_median_flipped' 'regenerated/numbers.json::piggyback.cfg_steer::cfg|b16::k_median_flipped' 'regenerated/numbers.json::cic.cfg_steer::cfg|b16::k_median_flipped' --digits 0
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_steer::concat|b16::flipped'
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_steer::concat|b16::n'
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_steer::concat|b1::flipped'
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_steer::concat|b1::n'
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_steer::concat|b16::flipped'
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_steer::concat|b16::n'
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_steer::concat|b1::flipped'
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_steer::concat|b1::n'
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_steer::concat|b16::flipped'
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_steer::concat|b16::n'
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_steer::concat|b1::flipped'
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_steer::concat|b1::n'
run "$PY" scripts/query.py value 'regenerated/numbers.json::cic.cfg_steer::concat|b16::flipped'
run "$PY" scripts/query.py value 'regenerated/numbers.json::cic.cfg_steer::concat|b16::n'
run "$PY" scripts/query.py value 'regenerated/numbers.json::cic.cfg_steer::concat|b1::flipped'
run "$PY" scripts/query.py value 'regenerated/numbers.json::cic.cfg_steer::concat|b1::n'
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_loc::cfg|identifier|clean::1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.cfg_loc::concat|identifier|clean::1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::proxy.colour::labeled|clean|1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_loc::cfg|identifier|clean::1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.cfg_loc::concat|identifier|clean::1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::malnet.colour::labeled|clean|1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_loc::cfg|identifier|clean::1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.cfg_loc::concat|identifier|clean::1048576::ratio' --digits 2
run "$PY" scripts/query.py value 'regenerated/numbers.json::piggyback.colour::labeled|clean|1048576::ratio' --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.cfg_loc::cfg|identifier|renamed::*::ratio' 'regenerated/numbers.json::malnet.cfg_loc::cfg|identifier|renamed::*::ratio' 'regenerated/numbers.json::piggyback.cfg_loc::cfg|identifier|renamed::*::ratio' 'regenerated/numbers.json::proxy.cfg_loc::concat|identifier|renamed::*::ratio' 'regenerated/numbers.json::malnet.cfg_loc::concat|identifier|renamed::*::ratio' 'regenerated/numbers.json::piggyback.cfg_loc::concat|identifier|renamed::*::ratio' --digits 2
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.cfg_loc::cfg|identifier|renamed::*::ratio' 'regenerated/numbers.json::malnet.cfg_loc::cfg|identifier|renamed::*::ratio' 'regenerated/numbers.json::piggyback.cfg_loc::cfg|identifier|renamed::*::ratio' 'regenerated/numbers.json::proxy.cfg_loc::concat|identifier|renamed::*::ratio' 'regenerated/numbers.json::malnet.cfg_loc::concat|identifier|renamed::*::ratio' 'regenerated/numbers.json::piggyback.cfg_loc::concat|identifier|renamed::*::ratio' --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.cfg_loc::cfg|fuzzy|clean::1048576::ratio' 'regenerated/numbers.json::proxy.cfg_loc::cfg|api|clean::1048576::ratio' 'regenerated/numbers.json::proxy.cfg_loc::cfg|combined|clean::1048576::ratio' 'regenerated/numbers.json::proxy.cfg_loc::cfg|structure|clean::1048576::ratio' 'regenerated/numbers.json::malnet.cfg_loc::cfg|fuzzy|clean::1048576::ratio' 'regenerated/numbers.json::malnet.cfg_loc::cfg|api|clean::1048576::ratio' 'regenerated/numbers.json::malnet.cfg_loc::cfg|combined|clean::1048576::ratio' 'regenerated/numbers.json::malnet.cfg_loc::cfg|structure|clean::1048576::ratio' 'regenerated/numbers.json::piggyback.cfg_loc::cfg|fuzzy|clean::1048576::ratio' 'regenerated/numbers.json::piggyback.cfg_loc::cfg|api|clean::1048576::ratio' 'regenerated/numbers.json::piggyback.cfg_loc::cfg|combined|clean::1048576::ratio' 'regenerated/numbers.json::piggyback.cfg_loc::cfg|structure|clean::1048576::ratio' --digits 2
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.cfg_loc::cfg|fuzzy|clean::1048576::ratio' 'regenerated/numbers.json::proxy.cfg_loc::cfg|api|clean::1048576::ratio' 'regenerated/numbers.json::proxy.cfg_loc::cfg|combined|clean::1048576::ratio' 'regenerated/numbers.json::proxy.cfg_loc::cfg|structure|clean::1048576::ratio' 'regenerated/numbers.json::malnet.cfg_loc::cfg|fuzzy|clean::1048576::ratio' 'regenerated/numbers.json::malnet.cfg_loc::cfg|api|clean::1048576::ratio' 'regenerated/numbers.json::malnet.cfg_loc::cfg|combined|clean::1048576::ratio' 'regenerated/numbers.json::malnet.cfg_loc::cfg|structure|clean::1048576::ratio' 'regenerated/numbers.json::piggyback.cfg_loc::cfg|fuzzy|clean::1048576::ratio' 'regenerated/numbers.json::piggyback.cfg_loc::cfg|api|clean::1048576::ratio' 'regenerated/numbers.json::piggyback.cfg_loc::cfg|combined|clean::1048576::ratio' 'regenerated/numbers.json::piggyback.cfg_loc::cfg|structure|clean::1048576::ratio' --digits 2
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.cfg_del::concat::lin_exact::flip_0.01 regenerated/numbers.json::malnet.cfg_del::concat::lin_exact::flip_0.01 regenerated/numbers.json::piggyback.cfg_del::concat::lin_exact::flip_0.01 regenerated/numbers.json::cic.cfg_del::concat::lin_exact::flip_0.01 --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.cfg_del::concat::lin_exact::flip_0.01 regenerated/numbers.json::malnet.cfg_del::concat::lin_exact::flip_0.01 regenerated/numbers.json::piggyback.cfg_del::concat::lin_exact::flip_0.01 regenerated/numbers.json::cic.cfg_del::concat::lin_exact::flip_0.01 --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.cfg_del::concat::lin_exact::flip_0.05 regenerated/numbers.json::malnet.cfg_del::concat::lin_exact::flip_0.05 regenerated/numbers.json::piggyback.cfg_del::concat::lin_exact::flip_0.05 regenerated/numbers.json::cic.cfg_del::concat::lin_exact::flip_0.05 --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.cfg_del::concat::lin_exact::flip_0.05 regenerated/numbers.json::malnet.cfg_del::concat::lin_exact::flip_0.05 regenerated/numbers.json::piggyback.cfg_del::concat::lin_exact::flip_0.05 regenerated/numbers.json::cic.cfg_del::concat::lin_exact::flip_0.05 --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.cfg_del::concat::lin_exact::flip_random_0.01 regenerated/numbers.json::malnet.cfg_del::concat::lin_exact::flip_random_0.01 regenerated/numbers.json::piggyback.cfg_del::concat::lin_exact::flip_random_0.01 regenerated/numbers.json::cic.cfg_del::concat::lin_exact::flip_random_0.01 --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.cfg_del::concat::lin_exact::flip_random_0.01 regenerated/numbers.json::malnet.cfg_del::concat::lin_exact::flip_random_0.01 regenerated/numbers.json::piggyback.cfg_del::concat::lin_exact::flip_random_0.01 regenerated/numbers.json::cic.cfg_del::concat::lin_exact::flip_random_0.01 --percent --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::proxy.cfg_del::cfg::lin_exact::flip_0.01 regenerated/numbers.json::malnet.cfg_del::cfg::lin_exact::flip_0.01 regenerated/numbers.json::piggyback.cfg_del::cfg::lin_exact::flip_0.01 regenerated/numbers.json::cic.cfg_del::cfg::lin_exact::flip_0.01 --percent --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::proxy.cfg_del::cfg::lin_exact::flip_0.01 regenerated/numbers.json::malnet.cfg_del::cfg::lin_exact::flip_0.01 regenerated/numbers.json::piggyback.cfg_del::cfg::lin_exact::flip_0.01 regenerated/numbers.json::cic.cfg_del::cfg::lin_exact::flip_0.01 --percent --digits 0
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::proxy.cfg_perturb::slack::identifier|*' 'regenerated/numbers.json::malnet.cfg_perturb::slack::identifier|*' 'regenerated/numbers.json::piggyback.cfg_perturb::slack::identifier|*' 'regenerated/numbers.json::cic.cfg_perturb::slack::identifier|*' --digits 1
run "$PY" scripts/query.py stat max 'regenerated/numbers.json::proxy.cfg_perturb::slack::identifier|*' 'regenerated/numbers.json::malnet.cfg_perturb::slack::identifier|*' 'regenerated/numbers.json::piggyback.cfg_perturb::slack::identifier|*' 'regenerated/numbers.json::cic.cfg_perturb::slack::identifier|*' --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.cfg_perturb::dmax --digits 0
run "$PY" scripts/query.py stat min regenerated/numbers.json::malnet.cfg_perturb::dmax regenerated/numbers.json::piggyback.cfg_perturb::dmax regenerated/numbers.json::cic.cfg_perturb::dmax --digits 0
run "$PY" scripts/query.py stat max regenerated/numbers.json::malnet.cfg_perturb::dmax regenerated/numbers.json::piggyback.cfg_perturb::dmax regenerated/numbers.json::cic.cfg_perturb::dmax --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.cfg_perturb::flip_cfg
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.cfg_perturb::flip_deployed
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.cfg_perturb::n --sep
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.cfg_perturb::flip_cfg
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.cfg_perturb::flip_deployed
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.cfg_perturb::n --sep
run "$PY" scripts/query.py value regenerated/numbers.json::cic.cfg_perturb::flip_cfg
run "$PY" scripts/query.py value regenerated/numbers.json::cic.cfg_perturb::flip_deployed
run "$PY" scripts/query.py value regenerated/numbers.json::cic.cfg_perturb::n --sep
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.cfg_perturb::flip_cfg
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.cfg_perturb::flip_deployed
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.cfg_perturb::n --sep
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.cfg_tree::cfg::k_cert_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.cfg_tree::concat::k_cert_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.cfg_tree::cfg::k_attack_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.cfg_tree::concat::k_attack_median --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.cfg_tree::cfg::k_attack_median --digits 0

# extended version: Results, separability across corpora (RQ4)
run cat regenerated/tables_extended/rq4_separability.tex
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.rq4::rho --percent --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.rq4::min_cut --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.rq4::rho --percent --digits 1
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.rq4::min_cut --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.rq4::rho --percent --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.rq4::min_cut --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.rq4::k_norm --permille --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.rq4::k_norm --permille --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.rq4::k_norm --permille --digits 0
run "$PY" scripts/query.py value regenerated/numbers.json::proxy.rq4::lift --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::malnet.rq4::lift --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::piggyback.rq4::lift --digits 2
run "$PY" scripts/query.py stat count 'regenerated/numbers.json::cic.payload_families::*'
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::n
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::apps::min_cut::spearman_with_k_norm --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::apps::min_cut::p --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::apps::rho::spearman_with_k_norm --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::apps::rho::p --digits 3
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::apps::boundary::spearman_with_k_norm --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::apps::boundary::p --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::family_medians::rho::n
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::family_medians::rho::spearman_with_k_norm --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::family_medians::min_cut::spearman_with_k_norm --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::family_medians::boundary::spearman_with_k_norm --digits 2
run "$PY" scripts/query.py stat min 'regenerated/numbers.json::rq4.pooled::family_medians::*::p' --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::within_family::rho::median_spearman --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::within_family::rho::n_positive
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::within_family::rho::n_families
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::within_family::min_cut::median_spearman --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::within_family::boundary::median_spearman --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::localization_family::rho::spearman_with_lift --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::localization_family::rho::p --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::localization_family::min_cut::spearman_with_lift --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::localization_family::min_cut::p --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::localization_family::boundary::spearman_with_lift --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::localization_family::k_norm::spearman_with_lift --digits 2
run "$PY" scripts/query.py value regenerated/numbers.json::rq4.pooled::localization_family::k_norm::p --digits 3

# extended version: Discussion, limitations
run "$PY" scripts/query.py group-size 'data/results/datasets/proxy_v3_gneardup_fcg/tier4/x2_method_level.json::apps::*' --by family --over-groups max
run "$PY" scripts/query.py group-size 'data/results/datasets/proxy_v3_gneardup_fcg/tier4/x2_method_level.json::apps::*' --by family --exclude-largest --over-groups min
run "$PY" scripts/query.py group-size 'data/results/datasets/proxy_v3_gneardup_fcg/tier4/x2_method_level.json::apps::*' --by family --exclude-largest --over-groups max

exit "$failed"
} 2>&1 | tee verify_output.txt
