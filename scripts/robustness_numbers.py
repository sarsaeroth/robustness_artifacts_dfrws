#!/usr/bin/env python3
"""Cross-corpus numbers and tables for the robustness paper (papers/robustness).

Every number the cross-corpus results text cites is read here from the result files and written to
regenerated/numbers.json; tables go to regenerated/tables_extended/*.tex and figures to
regenerated/figures/*.pdf (override with ROBUSTNESS_DATA and ROBUSTNESS_OUT). Re-run after any experiment finishes: missing results stay blank ("--").

Corpora under the common protocol (lineage grouping, FCG, one hash dimension):
  proxy      results/datasets/proxy_v3_gneardup_fcg   (corpus v3, one class per provider)
  malnet     results/datasets/malnet                  (MalNet families; lineage grouping is MalNet's default)
  piggyback  results/datasets/piggyback               (benign vs piggybacked)
  cic        results/datasets/cic_andmal              (CIC-AndMal2020 categories)
Payload-ground-truth analyses: proxy_v3_gneardup_fcg and <corpus>_payload.
"""
import json
import os
import sys

ART = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Result files (default: the artifact's data/results) and output directory (default: regenerated/).
DATA = os.environ.get("ROBUSTNESS_DATA", os.path.join(ART, "data", "results"))
OUT = os.environ.get("ROBUSTNESS_OUT", os.path.join(ART, "regenerated"))
RES = os.path.join(DATA, "datasets")
TAB = os.path.join(OUT, "tables_extended")
FIG = os.path.join(OUT, "figures")
NUM = os.path.join(OUT, "numbers.json")

CORPORA = ["proxy", "malnet", "piggyback", "cic"]
TAG = {"proxy": "proxy_v3_gneardup_fcg", "malnet": "malnet", "piggyback": "piggyback", "cic": "cic_andmal"}
PAYLOAD = {"proxy": "proxy_v3_gneardup_fcg", "malnet": "malnet_payload", "piggyback": "piggyback_payload",
           "cic": "cic_andmal_payload"}
LABEL = {"proxy": "Proxy", "malnet": "MalNet", "piggyback": "Piggyback", "cic": "CIC-AndMal2020"}
COLORINGS = [("labeled", "identifier", "\\textsc{identifier}"), ("combined", "combined", "\\textsc{combined}"),
             ("fuzzy", "fuzzy", "\\textsc{fuzzy}"), ("api", "api", "\\textsc{api}"),
             ("structure", "degree", "\\textsc{degree}")]
MODEL = {"additive": "additive", "pairwise": "pairwise", "gbdt": "GBDT", "rf": "RF", "svm_rbf": "RBF SVM",
         "mlp": "MLP"}

N = {}  # numbers.json


def load(tag, *parts):
    p = os.path.join(RES, tag, *parts)
    try:
        return json.load(open(p))
    except Exception:
        return None


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def put(key, val):
    N[key] = val
    return val


def table(name, header, rows, caption, label, star=False, size="\\small", colspec=None):
    if star:
        env = "table*"
    else:
        env = "table"
    cols = colspec or ("l" + "c" * (len(header) - 1))
    # the resizebox wrapper is used for single-column tables only
    if not star:
        resize_open = "\\resizebox{\\columnwidth}{!}{%"
        resize_close = "}"
    else:
        resize_open = ""
        resize_close = ""
    out = [f"\\begin{{{env}}}[t]", "\\centering", size, "\\setlength{\\tabcolsep}{4pt}",
           resize_open, f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
           " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        out.append(" & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}" + resize_close, f"\\caption{{{caption}}}",
            f"\\label{{{label}}}", f"\\end{{{env}}}"]
    os.makedirs(TAB, exist_ok=True)
    lines = []
    for x in out:
        if x:
            lines.append(x)
    open(os.path.join(TAB, name + ".tex"), "w").write("\n".join(lines) + "\n")


NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def protocol():
    """Protocol parameters that the captions state, from the experiment scripts, the result files and the manifests."""
    import ast
    import csv
    import math
    import re
    # module-level constants of the experiment scripts that set them
    consts = {}
    for script in ("exp_p3_frontier.py", "exp_bundle_attack.py", "exp_p3_steer_graph.py", "exp_l1_ladder.py",
                   "exp_p3_explainer_mx.py", "exp_p4_cfg_cert.py"):
        consts[script] = {}
        tree = ast.parse(open(os.path.join(ART, "scripts", "experiments", script)).read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    consts[script][node.targets[0].id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    put("protocol.d", consts["exp_p3_frontier.py"]["D"])
    put("protocol.cfg_d", consts["exp_p4_cfg_cert.py"]["D"])
    put("protocol.padding_budget", consts["exp_p3_frontier.py"]["MAX_T"])
    put("protocol.bundle_budget", consts["exp_bundle_attack.py"]["BUDGET"])
    put("protocol.stub_budget", consts["exp_p3_steer_graph.py"]["MAX_K"])
    put("protocol.flexible_models", len(consts["exp_l1_ladder.py"]["FLEXIBLE"]))
    put("protocol.deletion_fraction", min(consts["exp_p3_explainer_mx.py"]["KFRAC"]))
    # inflation sweep, as recorded in the proxy corpus's result file
    infl = load("proxy_v3", "exp_tier1_inflation.json")
    put("protocol.inflation_d", infl["dim"])
    put("protocol.inflation_k", max(infl["k_sweep"]))
    put("protocol.iso_dims", sorted(infl["iso_dims"]))
    # hash dimensions of the localization sweep, over the corpora with payload ground truth
    dims = set()
    for c in CORPORA:
        hd = load(PAYLOAD[c], "tier4", "x2_highdim.json") or {}
        for d in (hd.get("per_dim") or {}):
            dims.add(int(d))
    put("protocol.max_dim", max(dims))
    put("protocol.max_dim_log2", int(round(math.log2(max(dims)))))
    put("protocol.base_dim_log2", int(round(math.log2(min(dims)))))
    # forest size and smoothing parameters as recorded in the proxy corpus's result files
    model = g(load(TAG["proxy"], "tier4", "p3", "tree_cert.json"), "summary", "model")
    put("protocol.forest_trees", int(re.search(r"RandomForest\((\d+)\)", model).group(1)))
    sm = load("proxy_v3", "exp_tier3_certificates.json")
    put("protocol.smoothing", {"beta": sm["smoothing"]["deletion_prob"], "samples": sm["smoothing"]["n_samples"]})
    # Google Play applications left after the proxy-marker screen
    screened = 0
    for row in csv.DictReader(open(os.path.join(ART, "data", "manifests", "wild_play.csv"))):
        if row["proxy_marker_found"] == "0":
            screened += 1
    put("protocol.play_screened", screened)


def tex_int(x):
    """An integer in math mode with a thousands separator, e.g. 12{,}345."""
    return f"{x:,}".replace(",", "{,}")


# RQ1
def rq1():
    # linearity: tuned linear model against the best flexible model, per coloring
    rows = []
    for c in CORPORA:
        t = TAG[c]
        lad = load(t, "tier4", "l1_ladder_summary.json") or {}
        e2a = load(t, "tier4", "e2a_standardized.json") or {}
        tok = load(t, "tier4", "v2", "tok_analyses.json") or {}
        meta = load(t, "tier4", "v4", "h_ablation.json") or {}
        put(f"{c}.n", g(e2a, "labeled", "n"))
        put(f"{c}.classes", meta.get("classes"))
        if g(e2a, "labeled", "n"):
            n_cell = f"{g(e2a, 'labeled', 'n'):,}"
        else:
            n_cell = "--"
        row = [LABEL[c], str(meta.get("classes", "--")), n_cell]
        for rep, key, _ in COLORINGS[:3]:
            r = lad.get(rep) or {}
            lin = put(f"{c}.ladder.{key}.linear", g(r, "linear", "f1"))
            flex = {}
            for m, v in r.items():
                if m != "linear":
                    flex[m] = v
            if flex:
                # best flexible model by F1 (first one on ties)
                bm, bv = max(flex.items(), key=lambda kv: kv[1].get("f1", -1))
                put(f"{c}.ladder.{key}.best", {"model": bm, "f1": bv.get("f1"), "p": bv.get("p")})
                sig = bv.get("p") is not None and bv["p"] < 0.05 and bv["f1"] > (lin or 0)
                worse = []
                for m, v in flex.items():
                    if v.get("p") is not None and v["p"] < 0.05 and v["f1"] < (lin or 0):
                        worse.append(m)
                put(f"{c}.ladder.{key}.significantly_worse", sorted(worse))
                if sig:
                    star_mark = "$^{*}$"
                else:
                    star_mark = ""
                row += [f3(lin), f"{f3(bv.get('f1'))}{star_mark} ({MODEL[bm]})"]
            else:
                row += [f3(lin), "--"]
        hm = put(f"{c}.hardmargin_err.identifier", g(e2a, "labeled", "hardmargin_train_err"))
        nb = g(tok, "identifier", "L5", "naive_bayes")
        lg = g(tok, "identifier", "L5", "logistic")
        put(f"{c}.nb_vs_logistic.identifier", [nb, lg])
        row += [f3(hm), f"{f3(nb)} / {f3(lg)}"]
        rows.append(row)
    table("rq1_linearity",
          ["\\textbf{Corpus}", "\\textbf{Cl.}", "$n$",
           "\\textbf{Id.\\ lin.}", "\\textbf{Id.\\ best flex.}",
           "\\textbf{Comb.\\ lin.}", "\\textbf{Comb.\\ best flex.}",
           "\\textbf{Fuzzy lin.}", "\\textbf{Fuzzy best flex.}",
           "\\textbf{HM err.}", "\\textbf{NB / log.}"],
          rows,
          f"Linearity across corpora under the common protocol (lineage grouping, FCG, $d={tex_int(N['protocol.d'])}$). Macro-F1 of the "
          f"tuned linear model and of the best of {NUMBER_WORDS[N['protocol.flexible_models']]} flexible models (additive and pairwise boosting, gradient-boosted "
          "trees, random forest, RBF SVM, MLP) on the same folds; $^{*}$ the flexible model is significantly better "
          "(corrected resampled $t$-test, $p<0.05$). HM err.: training error of a hard-margin linear SVM on "
          "\\textsc{identifier} features. NB / log.: multinomial naive Bayes against logistic regression on exact "
          "\\textsc{identifier} tokens.",
          "tab:rq1", star=True)
    # refinement ablation
    rows = []
    for c in CORPORA:
        h = load(TAG[c], "tier4", "v4", "h_ablation.json") or {}
        row = [LABEL[c]]
        for _, key, name in COLORINGS:
            k = key
            v = h.get(k)
            if isinstance(v, dict) and "h0" in v:
                put(f"{c}.h.{k}", [v["h0"], v["h2"], v["p"]])
                if v["p"] < 0.05:
                    mark = "$^{*}$"
                else:
                    mark = ""
                row.append(f"{v['h0']:.3f} $\\to$ {v['h2']:.3f}{mark}")
            else:
                row.append("--")
        rows.append(row)
    header = ["\\textbf{Corpus}"]
    for _, _, n in COLORINGS:
        header.append(n)
    table("rq1_refinement", header, rows,
          "Macro-F1 of the tuned linear model on initial colors ($h=0$) and after two WL refinement rounds ($h=2$), "
          "FCG features, lineage grouping, two repetitions of five folds; $^{*}$ $p<0.05$ (corrected resampled $t$-test).",
          "tab:refinement", star=True)

    # representation comparison (paradigm) and margin/CKA/dimension diagnostics
    reps = [("labeled", "WL \\textsc{identifier}"), ("combined", "WL \\textsc{combined}"), ("fuzzy", "WL \\textsc{fuzzy}"),
            ("structure", "WL \\textsc{degree}"), ("mama_package", "MaMaDroid, package")]
    rows = []
    for rep, name in reps:
        row = [name]
        for c in CORPORA:
            e = g(load(TAG[c], "tier4", "e2a_standardized.json"), rep) or {}
            raw = e.get("lin_raw")
            std = e.get("lin_std")
            rf = e.get("rf")
            put(f"{c}.e2a.{rep}", {"raw": raw, "std": std, "rf": rf, "aniso": e.get("anisotropy_raw"),
                                   "hm": e.get("hardmargin_train_err")})
            row += [f"{f3(raw)} / {f3(std)}", f3(rf)]
        rows.append(row)
    hdr = ["\\textbf{Representation}"]
    for c in CORPORA:
        hdr += [f"\\textbf{{{LABEL[c]}}} lin.", "RF"]
    table("rq1_representations", hdr, rows,
          "Macro-F1 of a linear SGD model on raw and per-feature standardized features (raw / std) and of a random "
          "forest, per representation and corpus, under the common protocol. Standardization statistics come from "
          "the upsampled training fold.", "tab:paradigm", star=True)
    for c in CORPORA:
        lv = load(TAG[c], "tier4", "v3", "linearity_validate.json") or {}
        for rep in ("labeled", "combined", "fuzzy", "api", "structure"):
            put(f"{c}.cka.{rep}", g(lv, "B7", rep, "cka"))
            put(f"{c}.margin.{rep}", {"p10": g(lv, "B8", rep, "margin_p10"), "within": g(lv, "B8", rep, "within_margin_frac")})
        put(f"{c}.dim_scaling", g(lv, "B9"))
        put(f"{c}.synthetic", g(lv, "B6"))
    # renaming
    for c in CORPORA:
        r = load(TAG[c], "tier4", "r3a_renaming.json") or {}
        put(f"{c}.renaming.flip_rate", g(r, "labeled", "flip_rate"))
        put(f"{c}.renaming.clean_accuracy", g(r, "labeled", "clean_accuracy"))


# RQ2
IDTAG = {"proxy": "proxy_v3", "malnet": "malnet", "piggyback": "piggyback", "cic": "cic_andmal"}


def _ratio(fam_dict, field="p_at_target"):
    """median over families (base rate > 0) of precision / base rate"""
    import statistics
    rs = []
    for f, v in fam_dict.items():
        if isinstance(v, dict) and v.get("base_rate") and v.get(field) is not None:
            rs.append(v[field] / v["base_rate"])
    if rs:
        return statistics.median(rs)
    return None


def rq2():
    rows = []
    for c in CORPORA:
        pt = PAYLOAD[c]
        ident = load(IDTAG[c], "exp_tier1_identifiability.json") or {}
        infl = load(IDTAG[c], "exp_tier1_inflation.json") or {}
        x2 = load(pt, "tier4", "x2_method_level.json") or {}
        hd = load(pt, "tier4", "x2_highdim.json") or {}
        mx = load(pt, "tier4", "p3", "explainer_mx.json") or {}
        cl = load(pt, "tier4", "p3", "colour_localize.json") or {}
        gn = load(pt, "tier4", "g2_gnn_localize.json") or {}
        rn = load(pt, "tier4", "p3", "renorm.json") or {}
        lt = load(pt, "tier4", "v3", "localize_truth.json") or {}
        fams = sorted(g(x2, "summary") or {})
        put(f"{c}.payload_families", fams)

        # identifiability and inflation
        e1 = g(ident, "E1_isolation_vs_prediction") or {}
        abs_err = []
        for v in e1.values():
            abs_err.append(v.get("max_abs_error", 0))
        put(f"{c}.ident.max_abs_error", max(abs_err, default=None))
        signed_err = []
        for v in e1.values():
            signed_err.append(abs(v.get("median_signed_error", 0)))
        put(f"{c}.ident.median_signed_error_max", max(signed_err, default=None))
        put(f"{c}.ident.rho", g(ident, "E2_occupancy_ratio"))
        put(f"{c}.ident.colouring_M", g(ident, "E4_colouring_vs_M_distinct"))
        r0 = g(infl, "rows", "0") or {}
        # the largest number of inserted methods in the sweep
        if infl.get("k_sweep"):
            k_top = str(max(infl["k_sweep"]))
        else:
            k_top = None
        rk = g(infl, "rows", k_top) or {}
        put(f"{c}.infl", {"apps": infl.get("apps"), "iso0": r0.get("isolation_by_dim"), "isok": rk.get("isolation_by_dim"),
                          "margin_ratio": rk.get("margin_ratio"), "still_correct": rk.get("still_correct_frac")})

        # localization at the common hash dimension, per explainer and per family
        summ = g(x2, "summary") or {}
        ex = {}
        for e in (x2.get("explainers") or []):
            per_fam = {}
            for f in summ:
                per_fam[f] = g(summ, f, e)
            ex[e] = _ratio(per_fam)
        put(f"{c}.loc1024.ratio_by_explainer", ex)
        base_by_family = {}
        for f in summ:
            base_by_family[f] = g(summ, f, "lin_exact", "base_rate")
        put(f"{c}.loc1024.base_by_family", base_by_family)
        exact_by_family = {}
        for f in summ:
            exact_by_family[f] = g(summ, f, "lin_exact", "p_at_target")
        put(f"{c}.loc1024.exact_by_family", exact_by_family)

        # localization by hash dimension
        dims = sorted((g(hd, "per_dim") or {}), key=int)
        ratio_by_dim = {}
        for d in dims:
            ratio_by_dim[d] = _ratio(hd["per_dim"][d])
        put(f"{c}.highdim.ratio_by_dim", ratio_by_dim)
        if dims:
            p_by_family = {}
            for f in (g(hd, "per_dim", dims[-1]) or {}):
                p_by_family[f] = g(hd, "per_dim", dims[-1], f, "p_at_target")
        else:
            p_by_family = None
        put(f"{c}.highdim.p_by_family_maxdim", p_by_family)
        mx_summary = {}
        for e in (g(mx, "summary") or {}):
            mx_summary[e] = {}
            for k, v in (g(mx, "summary", e) or {}).items():
                mx_summary[e][k] = v
        put(f"{c}.mx", mx_summary)

        # colorings, clean and renamed
        col = {}
        for key in ("labeled", "fuzzy", "api", "structure"):
            for state in ("clean", "renamed"):
                for d, v in (cl.get(f"{key}|{state}") or {}).items():
                    per_fam = {}
                    for f, x in v.items():
                        if f != "accuracy":
                            per_fam[f] = x
                    col[f"{key}|{state}|{d}"] = {"accuracy": v.get("accuracy"), "ratio": _ratio(per_fam)}
        put(f"{c}.colour", col)
        granularity = {}
        for k, v in (cl.get("granularity") or {}).items():
            granularity[k] = {}
            for f in v:
                granularity[k][f] = g(v, f, "sdk_colour_shared_with_host")
        put(f"{c}.granularity", granularity)
        gnn_fams = {}
        for f in (gn.get("base_rate") or {}):
            gnn_fams[f] = {"p_at_target": g(gn, "localize_p_at_target", f), "base_rate": g(gn, "base_rate", f)}
        put(f"{c}.gnn", {"macro_f1": gn.get("macro_f1"), "ratio": _ratio(gnn_fams)})
        put(f"{c}.renorm", g(rn, "median"))
        put(f"{c}.seed_spearman", g(lt, "D19"))

        # SDK transplant
        if c == "proxy":
            tp_tag = "proxy_v3_gneardup"
        else:
            tp_tag = pt
        tp = load(tp_tag, "tier4", "v3", "sdk_transplant.json") or {}
        if "A2a" in tp:
            a2a = {}
            for k in ("full", "sdk", "host"):
                a2a[k] = g(tp, "A2a", k, "macro_f1")
            a2b = {}
            for th, v in tp["A2b"].items():
                a2b[th] = {}
                for k in ("full", "sdk", "host"):
                    a2b[th][k] = g(v, k, "macro_f1")
            a2c = {}
            for w in ("unseen", "seen"):
                a2c[w] = g(tp, "A2c", "dex_grouped", "all", w)
            put(f"{c}.transplant", {"token_frac": tp.get("median_sdk_token_frac"),
                                    "A2a": a2a,
                                    "A2a_sdk_vs_host_p": g(tp, "A2a", "sdk_minus_host", "p"),
                                    "A2b": a2b,
                                    "A2c": a2c})
        else:
            put(f"{c}.transplant", {"skipped": tp.get("skipped")})
        auroc_by_family = {}
        for f, v in (g(lt, "D18") or {}).items():
            auroc_by_family[f] = g(v, "auroc")
        put(f"{c}.auroc_by_family", auroc_by_family)

        # table row
        if dims:
            top = max(dims, key=int)
        else:
            top = None
        def fr(x):
            if x is None:
                return "--"
            return f"{x:.2f}$\\times$"
        mlp = ex.get("mlp_ig")
        tree_ratios = []
        for k, v in ex.items():
            if k.startswith(("rf", "xgb")) and v is not None:
                tree_ratios.append(v)
        tree = max(tree_ratios, default=None)
        if ex.get("lin_exact") is None:
            rows.append([LABEL[c], "0", "\\multicolumn{8}{c}{no payload ground truth: localization undefined}"])
            continue
        # cells use `a and b`: a falsy first operand is passed to fr unchanged
        if top:
            exact_top = g(load(pt, "tier4", "x2_highdim.json") or {}, "per_dim")
            if exact_top:
                exact_top = _ratio(hd["per_dim"][top])
            exact_top_cell = fr(exact_top)
        else:
            exact_top_cell = "--"
        if mx:
            flip_key = f"M3_flip_{N['protocol.deletion_fraction']}"
            flip_cell = f"{(g(mx, 'summary', 'lin_exact', flip_key) or 0):.2f}"
        else:
            flip_cell = "--"
        if top:
            fuzzy_cell = fr(g(col, f"fuzzy|clean|{top}", "ratio"))
        else:
            fuzzy_cell = "--"
        if top:
            renamed_cell = fr(g(col, f"labeled|renamed|{top}", "ratio"))
        else:
            renamed_cell = "--"
        gin = g(load(pt, "tier4", "g2_gnn_localize.json") or {}, "base_rate")
        if gin:
            gin_fams = {}
            for f in gn["base_rate"]:
                gin_fams[f] = {"p_at_target": g(gn, "localize_p_at_target", f), "base_rate": g(gn, "base_rate", f)}
            gin = _ratio(gin_fams)
        gin_cell = fr(gin)
        rows.append([LABEL[c], str(len(fams)), fr(ex.get("lin_exact")), fr(tree), fr(mlp),
                     exact_top_cell, flip_cell, fuzzy_cell, renamed_cell, gin_cell])
    top_dim = "$2^{" + str(N["protocol.max_dim_log2"]) + "}$"
    deleted = f"{100 * N['protocol.deletion_fraction']:.0f}\\%"
    table("rq2_localization",
          ["\\textbf{Corpus}", "\\textbf{Fam.}", "\\textbf{Exact}", "\\textbf{Best tree}", "\\textbf{MLP IG}",
           f"\\textbf{{Exact, {top_dim}}}", f"\\textbf{{Flip, {deleted}}}", f"\\textbf{{\\textsc{{fuzzy}}, {top_dim}}}",
           f"\\textbf{{Renamed, {top_dim}}}", "\\textbf{GIN exact}"],
          rows,
          "Method-level localization against payload ground truth on the four corpora: precision at $|$target$|$ "
          "divided by the base rate, median over the families with ground truth (Fam.). Exact: Proposition~\\ref{prop:methodshap} "
          f"at $d={tex_int(N['protocol.d'])}$; Best tree: the better of path and interventional TreeSHAP on a random forest and on boosting; "
          f"MLP IG: Integrated Gradients; Exact, {top_dim}: the exact decomposition at $d={top_dim[1:-1]}$; Flip, {deleted}: share of "
          f"predictions changed by deleting the top-ranked {deleted} of methods and re-extracting the graph; \\textsc{{fuzzy}}, "
          f"{top_dim}: the exact decomposition under the renaming-invariant \\textsc{{fuzzy}} coloring; Renamed: "
          "\\textsc{identifier} after per-application renaming; GIN exact: exact mean-pool decomposition of a three-layer GIN.",
          "tab:rq2", star=True)


def fig_localization():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    colors = {"proxy": "#2a78d6", "malnet": "#eb6834", "piggyback": "#1baf7a"}
    for c in ("proxy", "malnet", "piggyback"):
        hd = load(PAYLOAD[c], "tier4", "x2_highdim.json") or {}
        dims = sorted(hd.get("per_dim", {}), key=int)
        ys = []
        for d in dims:
            ys.append(_ratio(hd["per_dim"][d]))
        xs = []
        for d in dims:
            xs.append(math.log2(int(d)))
        ax.plot(xs, ys, marker="o", ms=3, lw=2, color=colors[c], label=LABEL[c])
    ax.axhline(1, color="#52514e", lw=1, ls="--")
    ax.text(19.9, 0.7, "base rate", fontsize=6.5, color="#52514e", ha="right")
    ax.set_xlabel("hash dimension $\\log_2 d$", fontsize=8)
    ax.set_ylabel("precision / base rate", fontsize=8)
    ax.set_ylim(0, 6)
    ax.set_xticks(range(10, 21, 2))
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "localization_corpora.pdf"))
    plt.close(fig)


def tab_inflation():
    rows = []
    for c in ("proxy", "malnet", "piggyback"):
        f = N.get(f"{c}.infl") or {}
        i0 = f.get("iso0") or {}
        ik = f.get("isok") or {}
        row = [LABEL[c], str(f.get("apps", "--"))]
        for dim in N["protocol.iso_dims"]:
            d = str(dim)
            if d in i0:
                row.append(f"{i0.get(d, float('nan')):.3f} $\\to$ {ik.get(d, float('nan')):.3f}")
            else:
                row.append("--")
        if f.get("margin_ratio") is not None:
            row.append(f"{f.get('margin_ratio'):.2f}")
        else:
            row.append("--")
        rows.append(row)
    import math
    header = ["\\textbf{Corpus}", "\\textbf{Apps}"]
    for dim in N["protocol.iso_dims"]:
        header.append("$2^{" + str(int(round(math.log2(dim)))) + "}$")
    header.append("\\textbf{Margin kept}")
    table("rq2_inflation", header, rows,
          f"Fraction of payload tokens in pure buckets before and after inserting $k={tex_int(N['protocol.inflation_k'])}$ inert methods, by hash "
          f"dimension, and the share of the mean prediction margin retained at $d={tex_int(N['protocol.inflation_d'])}$. CIC-AndMal2020 has no payload "
          "ground truth.", "tab:inflation", star=True)


# RQ3
FRONT = [("labeled", "identifier"), ("combined", "combined"), ("fuzzy", "fuzzy"), ("api", "api"), ("structure", "degree")]


def rq3():
    rows = []
    for c in CORPORA:
        t = TAG[c]
        fr = load(t, "tier4", "p3", "frontier.json") or {}
        r = [LABEL[c]]
        # accuracy / certificate / attack frontier per coloring
        for key, name in FRONT:
            v = fr.get(f"sgd_{key}") or {}
            frontier = {}
            for k in ("f1", "n_cert", "k_cert_median", "k_cert_q", "k_steer_median", "steer_over_cert_median",
                      "rand_never_frac"):
                frontier[k] = v.get(k)
            put(f"{c}.frontier.{name}", frontier)
        for key, name in (("labeled", "identifier"), ("combined", "combined"), ("fuzzy", "fuzzy"), ("structure", "degree")):
            v = fr.get(f"sgd_{key}") or {}
            if v:
                r.append(f"{v['f1']:.3f} / {v['k_cert_median']:.0f}--{v['k_steer_median']:.0f}")
            else:
                r.append("--")
        if g(fr, "sgd_labeled", "rand_never_frac") is not None:
            r.append(f"{g(fr, 'sgd_labeled', 'rand_never_frac'):.2f}")
        else:
            r.append("--")
        lin = fr.get("linf") or {}
        linf = {}
        for k, v in lin.items():
            if isinstance(v, dict):
                linf[k] = {}
                for kk in ("f1", "k_cert_median", "k_steer_median"):
                    linf[k][kk] = v.get(kk)
            else:
                linf[k] = v
        put(f"{c}.linf", linf)
        mono = fr.get("monotone") or {}
        monotone = {}
        for f, v in mono.items():
            if isinstance(v, dict):
                monotone[f] = {}
                for k in ("auc_monotone", "auc_unconstrained"):
                    monotone[f][k] = v.get(k)
        put(f"{c}.monotone", monotone)

        # forest certificate, Drebin and MaMaDroid certificates
        tc = g(load(t, "tier4", "p3", "tree_cert.json"), "summary") or {}
        tree_cert = {}
        for k in ("n", "k_cert_median", "k_attack_median", "zero_cert_frac"):
            tree_cert[k] = tc.get(k)
        put(f"{c}.tree_cert", tree_cert)
        if tc:
            r.append(f"{tc.get('k_cert_median', 0):.0f} / {tc.get('k_attack_median', 0):.0f}")
        else:
            r.append("--")
        dr = g(load(t, "tier4", "v4", "cert_drebin.json"), "drebin_code") or {}
        drebin_cert = {}
        for m in ("linear_sgd", "linear_svm"):
            drebin_cert[m] = {}
            for k in ("f1", "k_median", "unconditional"):
                drebin_cert[m][k] = g(dr, m, k)
        put(f"{c}.drebin_cert", drebin_cert)
        if g(dr, "linear_svm", "f1"):
            r.append(f"{g(dr, 'linear_svm', 'f1'):.3f} / {g(dr, 'linear_svm', 'k_median'):.0f}")
        else:
            r.append("--")
        l1 = load(t, "tier4", "v4", "cert_l1block.json") or {}
        mama_cert = {}
        for m in ("mama_family", "mama_family_std", "mama_package", "mama_package_std"):
            mama_cert[m] = {}
            for k in ("f1", "k_median_when_finite", "unconditional"):
                mama_cert[m][k] = g(l1, m, k)
        put(f"{c}.mama_cert", mama_cert)
        rows.append(r)

        # certificate tightness, theorem bounds, attack realism
        ct = load(t, "tier4", "v3", "cert_tightness.json") or {}
        tightness = {}
        for col in ("labeled", "fuzzy", "api", "structure"):
            tightness[col] = g(ct, "C11_C12", col, "overall")
        put(f"{c}.tightness", tightness)
        put(f"{c}.salt_spearman", g(ct, "C14", "spearman_median"))
        tb = load(t, "tier4", "p3", "thm_bounds.json") or {}
        bound_summary = {}
        for k, v in (tb.get("summary") or {}).items():
            bound_summary[k] = {}
            for kk in ("L_le_bound_global_frac", "log10_slack_global_median", "log10_slack_local_median", "L_per_op_median"):
                bound_summary[k][kk] = v.get(kk)
        put(f"{c}.thm_bounds", {"apps": tb.get("apps"), "dmax": tb.get("dmax_global_median"), "summary": bound_summary})
        ar = load(t, "tier4", "v3", "attack_realism.json") or {}
        realism = {}
        for col in ("labeled", "structure"):
            realism[col] = g(ar, "results", col, "_overall")
        put(f"{c}.realism", realism)

        # graph perturbations (r4_perturb.jsonl): flip rates and retention, bounded edits and flattening
        p = os.path.join(RES, TAG[c], "tier4", "r4_perturb.jsonl")
        if not os.path.exists(p):
            r4 = None
        else:
            import statistics
            records = []
            for l in open(p):
                records.append(json.loads(l))
            per = []
            for rec in records:
                if "op" in rec and "error" not in rec:
                    per.append(rec)
            shas = set()
            for rec in per:
                shas.add(rec["sha"])
            r4 = {"n": len(per), "apps": len(shas)}
            bounded = []
            for rec in per:
                if rec["op"] != "flatten":
                    bounded.append(rec)
            for col in ("labeled", "fuzzy", "api", "structure"):
                fl = []
                for rec in bounded:
                    if col in rec["col"]:
                        fl.append(rec["col"][col]["flip"])
                if fl:
                    r4[f"flip_{col}"] = sum(fl) / len(fl)
                else:
                    r4[f"flip_{col}"] = None
                for loc in ("sdk", "host"):
                    ret = []
                    for rec in bounded:
                        if rec["locale"] == loc and col in rec["col"] and rec["col"][col]["retention"] is not None:
                            ret.append(rec["col"][col]["retention"])
                    if ret:
                        r4[f"ret_{col}_{loc}"] = statistics.mean(ret)
                    else:
                        r4[f"ret_{col}_{loc}"] = None
                for loc in ("sdk", "host", "all"):
                    fl = []
                    for rec in per:
                        if rec["op"] == "flatten" and rec["locale"] == loc and col in rec["col"]:
                            fl.append(rec["col"][col]["flip"])
                    if fl:
                        r4[f"flatten_{col}_{loc}"] = sum(fl) / len(fl)
                    else:
                        r4[f"flatten_{col}_{loc}"] = None
        put(f"{c}.r4", r4)

        # renaming soundness
        r3 = g(load(t, "tier4", "r3a_renaming.json"), "labeled") or {}
        soundness = {}
        for k in ("n_evals", "flip_rate", "inside_certified_frac", "unsound_flips_inside", "delta_over_radius_median",
                  "delta_over_radius_q", "radius_median"):
            soundness[k] = r3.get(k)
        put(f"{c}.renaming_soundness", soundness)
    table("rq3_frontier",
          ["\\textbf{Corpus}", "\\textsc{identifier}", "\\textsc{combined}", "\\textsc{fuzzy}", "\\textsc{degree}",
           "\\textbf{Rand.\\ never}", "\\textbf{RF cert / attack}", "\\textbf{Drebin SVM}"],
          rows,
          f"Accuracy and certified robustness to code addition on the four corpora (lineage grouping, FCG, $d={tex_int(N['protocol.d'])}$). "
          "For each coloring: macro-F1 / median additive certificate (Theorem~\\ref{thm:certadd}) in added tokens over "
          "staged applications held out of each fold, and the median budget of the greedy aimed attack (certificate--attack). "
          f"Rand.\\ never: share of \\textsc{{identifier}} applications that random padding never flips within {N['protocol.padding_budget']:,} tokens. "
          f"RF: interval-propagation certificate and greedy attack for a {N['protocol.forest_trees']}-tree forest. Drebin SVM: macro-F1 / exact "
          "median certificate in added features for code-only Drebin features with a linear SVM.",
          "tab:rq3", star=True)


def fig_frontier():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"proxy": "#2a78d6", "malnet": "#eb6834", "piggyback": "#1baf7a", "cic": "#eda100"}
    markers = {"identifier": "o", "combined": "s", "fuzzy": "^", "api": "v", "degree": "D"}
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for c in CORPORA:
        fr = load(TAG[c], "tier4", "p3", "frontier.json") or {}
        for key, name in FRONT:
            v = fr.get(f"sgd_{key}")
            if not v or not v.get("k_steer_median"):
                continue
            ax.scatter([max(v["k_steer_median"], 1)], [v["f1"]], color=colors[c], marker=markers[name], s=22,
                       edgecolors="white", linewidths=0.5, zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlim(5, 3000)
    ax.set_xlabel("median aimed-attack budget (added tokens, log scale)", fontsize=7.5)
    ax.set_ylabel("macro-F1", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    # legend: corpora by color, colorings by marker
    handles = []
    for c in CORPORA:
        handles.append(Line2D([], [], color=colors[c], marker="o", ls="", label=LABEL[c]))
    for n, m in markers.items():
        handles.append(Line2D([], [], color="#52514e", marker=m, ls="", label=n))
    ax.legend(handles=handles, frameon=False, fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "frontier_corpora.pdf"))
    plt.close(fig)


# open world
def rq_open():
    rows = []
    for c in CORPORA:
        t = TAG[c]
        o = load(t, "tier4", "v3", "open_world.json") or {}
        o1 = {}
        for m, v in (o.get("O1") or {}).items():
            o1[m] = v
        put(f"{c}.open.O1", o1)
        o2 = {}
        for m, v in (o.get("O2") or {}).items():
            o2[m] = v
        put(f"{c}.open.O2", o2)
        put(f"{c}.open.abstain", o.get("O1_abstain"))
        o3 = o.get("O3") or {}
        o3_vals = {}
        for k in ("n", "k_any_median", "k_benign_median", "attack_any_median", "attack_benign_median",
                  "untargeted_lands_benign"):
            o3_vals[k] = o3.get(k)
        put(f"{c}.open.O3", o3_vals)
        if c == "proxy":
            bundle_tag = "proxy_v3_gneardup"
        else:
            bundle_tag = t
        b = load(bundle_tag, "tier4", "v3", "bundle_attack.json") or {}
        bundle = {}
        for k in ("n", "n_packages", "token", "greedy", "random", "greedy_packages_median", "whole_app_flip_rate_mean"):
            bundle[k] = b.get(k)
        put(f"{c}.bundle", bundle)
        # Google Play results per coloring, if released
        if c == "proxy":
            wild_tag = "proxy_v3"
        else:
            wild_tag = t
        wc = None
        pth = os.path.join(DATA, "cross_dataset", f"wild_colorings_{wild_tag}_gneardup_fcg.json")
        if os.path.exists(pth):
            wc = json.load(open(pth))
        put(f"{c}.wild_colorings", wc)
        ab = o.get("O1_abstain") or {}
        lin = g(o, "O2", "linear", "macro_f1")
        rf = g(o, "O2", "rf", "macro_f1")
        fp_lin = g(o, "O1", "linear", "wild_fp_rate")
        fp_rf = g(o, "O1", "rf", "wild_fp_rate")
        if "auroc_proxy_vs_wild" in ab:
            abstain_cell = f"{f3(ab.get('auroc_proxy_vs_wild'))} / {f3(ab.get('wild_fp_at_tau'))}"
        else:
            abstain_cell = "n/a"
        if b:
            bundle_cell = f"{g(b, 'greedy', 'success'):.2f} / {g(b, 'greedy', 'median_tokens_when_successful'):,.0f}"
        else:
            bundle_cell = "--"
        rows.append([LABEL[c], str(len(o.get("classes") or [])),
                     f"{f3(lin)} / {f3(rf)}", f"{f3(fp_lin)} / {f3(fp_rf)}",
                     abstain_cell,
                     f"{o3.get('untargeted_lands_benign', 0):.2f}", f"{o3.get('attack_benign_median', 0):.0f}",
                     bundle_cell])
    table("open_world",
          ["\\textbf{Corpus}", "\\textbf{Cl.}", "\\textbf{F1 lin. / RF}", "\\textbf{Play FP lin. / RF}",
           "\\textbf{Abstain AUROC / FP@95}", "\\textbf{Lands benign}", "\\textbf{To benign}", "\\textbf{Bundling}"],
          rows,
          f"The open world on the four corpora (lineage grouping, FCG, $d={tex_int(N['protocol.d'])}$). F1: macro-F1 with the corpus's benign "
          f"class as an extra class; Play FP: share of the {N['protocol.play_screened']:,} marker-screened Google Play applications assigned a "
          "non-benign class; Abstain: model without the benign class, AUROC of known against Play applications by maximum "
          "class probability and the share of Play applications above the threshold that keeps 95\\% of known ones "
          "(undefined with one non-benign class); Lands benign: share of untargeted aimed attacks that end in the benign "
          "class; To benign: median tokens for the attack aimed at the benign class; Bundling: success rate and median "
          f"added tokens of greedily bundling whole unused packages from benign applications, within {N['protocol.bundle_budget']:,} tokens.",
          "tab:open", star=True)


# CFG
CFGTAG = {"proxy": "proxy_v3_gneardup", "malnet": "malnet", "piggyback": "piggyback", "cic": "cic_andmal"}
CFGPAY = {"proxy": "proxy_v3_gneardup", "malnet": "malnet_payload", "piggyback": "piggyback_payload", "cic": "cic_andmal_payload"}
CFGPERT = {"proxy": "proxy_v3_gneardup", "malnet": "malnet_cfg+fcg", "piggyback": "piggyback_cfg+fcg", "cic": "cic_andmal_cfg+fcg"}


def rq_cfg():
    rows = []
    for c in CORPORA:
        cc = load(CFGTAG[c], "tier4", "p4", "cfg_cert.json") or {}
        cfg_cert = {}
        for k, v in cc.items():
            cfg_cert[k] = {}
            for kk in ("f1", "n", "k_certgrad_median", "k_block_median", "k_attack_median", "violations",
                       "attack_over_cert_median"):
                cfg_cert[k][kk] = v.get(kk)
        put(f"{c}.cfg_cert", cfg_cert)
        st = load(CFGTAG[c], "tier4", "p4", "cfg_steer.json") or {}
        cfg_steer = {}
        for k, v in st.items():
            cfg_steer[k] = {}
            for kk in ("n", "flipped", "k_median_flipped", "never_frac"):
                cfg_steer[k][kk] = v.get(kk)
        put(f"{c}.cfg_steer", cfg_steer)
        lo = load(CFGPAY[c], "tier4", "p4", "cfg_localize.json") or {}
        cfg_loc = {}
        for k, dv in lo.items():
            if "|" in k:
                cfg_loc[k] = {}
                for d, v in dv.items():
                    per_fam = {}
                    for f, x in v.items():
                        if f != "accuracy":
                            per_fam[f] = x
                    cfg_loc[k][d] = {"accuracy": v.get("accuracy"), "ratio": _ratio(per_fam)}
        put(f"{c}.cfg_loc", cfg_loc)
        de = load(CFGPAY[c], "tier4", "p4", "cfg_deletion.json") or {}
        cfg_del = {}
        for m in ("cfg", "concat"):
            cfg_del[m] = g(de, m, "summary")
        put(f"{c}.cfg_del", cfg_del)
        pe = load(CFGPERT[c], "tier4", "p4", "cfg_perturb.json") or {}
        fl = pe.get("flips") or []
        flip_cfg = 0
        for f in fl:
            if f.get("flip_cfg"):
                flip_cfg += 1
        flip_deployed = 0
        for f in fl:
            if f.get("flip_deployed"):
                flip_deployed += 1
        slack = {}
        for k, v in (pe.get("summary") or {}).items():
            if isinstance(v, dict):
                slack[k] = v.get("log10_slack_local")
        put(f"{c}.cfg_perturb", {"dmax": g(pe, "summary", "dmax_median"), "n": len(fl),
                                 "flip_cfg": flip_cfg, "flip_deployed": flip_deployed,
                                 "slack": slack})
        tr = load(CFGTAG[c], "tier4", "p4", "cfg_tree.json") or {}
        cfg_tree = {}
        for m in ("cfg", "concat"):
            cfg_tree[m] = {}
            for kk in ("f1", "k_cert_median", "zero_frac", "k_attack_median"):
                cfg_tree[m][kk] = g(tr, m, kk)
        put(f"{c}.cfg_tree", cfg_tree)
        def cell(key):
            # "F1 / certificate--attack" for one model, "degenerate" for a certificate of 10^4 or more
            v = cc.get(key) or {}
            if not v or v.get("f1") is None:
                return "--"
            if v.get("k_certgrad_median") is not None:
                k = v.get("k_certgrad_median")
            else:
                k = v.get("k_block_median")
            a = v.get("k_attack_median")
            if k is not None and k >= 1e4:
                return f"{v['f1']:.3f} / degenerate"
            if a:
                return f"{v['f1']:.3f} / {k:,.0f}--{a:,.0f}"
            return f"{v['f1']:.3f} / {k:,.0f}"
        rows.append([LABEL[c], cell("fcg_block|identifier"), cell("cfg|identifier"), cell("concat|identifier"),
                     cell("cfg|combined"), cell("concat|combined")])
    table("cfg_deployed",
          ["\\textbf{Corpus}", "\\textbf{FCG id.}", "\\textbf{CFG id.}", "\\textbf{Deployed id.}",
           "\\textbf{CFG comb.}", "\\textbf{Deployed comb.}"],
          rows,
          f"FCG, CFG and the deployed CFG and FCG classifier on the four corpora (lineage grouping, $d={tex_int(N['protocol.cfg_d'])}$ per graph): "
          "macro-F1 / median certificate in added tokens (Theorem~\\ref{thm:certgrad} for single-graph models, the "
          "block-normalized Theorem~\\ref{thm:certblock} for the deployed model)--median greedy aimed attack.",
          "tab:cfg", star=True)


# RQ4
RQ4TAG = {"proxy": "proxy_v3_payload_gneardup_fcg", "malnet": "malnet_payload", "piggyback": "piggyback_payload",
          "cic": "cic_andmal_payload"}


def rq4():
    import statistics as st
    pooled = json.load(open(os.path.join(DATA, "cross_dataset", "rq4_pooled.json")))
    put("rq4.pooled", pooled)
    rows = []
    for c in CORPORA:
        r = (load(RQ4TAG[c], "tier4", "v4", "rq4.json") or {}).get("rows") or []
        if not r:
            rows.append([LABEL[c], "0", "\\multicolumn{5}{c}{no payload ground truth}"])
            continue

        def med(k):
            # median of field k over the staged applications that have it
            vals = []
            for x in r:
                if x.get(k) is not None:
                    vals.append(x[k])
            return st.median(vals)
        # localization lift; `a and b`: a falsy summary is passed to _ratio unchanged
        lift_input = g(load(PAYLOAD[c], "tier4", "x2_method_level.json") or {}, "summary")
        if lift_input:
            lift_input = {}
            for f, v in load(PAYLOAD[c], "tier4", "x2_method_level.json")["summary"].items():
                lift_input[f] = v.get("lin_exact")
        lift = _ratio(lift_input)
        families = set()
        for x in r:
            families.add(x["family"])
        put(f"{c}.rq4", {"n": len(r), "families": len(families), "rho": med("rho"), "min_cut": med("min_cut"),
                         "boundary": med("boundary"), "k_norm": med("k_norm"), "lift": lift})
        if lift:
            lift_cell = f"{lift:.2f}$\\times$"
        else:
            lift_cell = "--"
        rows.append([LABEL[c], str(len(r)), f"{med('rho'):.3f}", f"{med('min_cut'):.0f}", f"{med('boundary'):.3f}",
                     f"{1000 * med('k_norm'):.1f}", lift_cell])
    table("rq4_separability",
          ["\\textbf{Corpus}", "\\textbf{Apps}", "$\\rho$", "\\textbf{Min cut}", "\\textbf{Boundary}",
           "\\textbf{Cert.\\ / 1k tokens}", "\\textbf{Loc.\\ lift}"],
          rows,
          "Payload separability on the staged payload applications of each corpus, medians: occupancy $\\rho$ (payload "
          "share of non-library methods), minimum edge cut between host entry methods and the payload, boundary edges per "
          "payload method, the additive \\textsc{identifier} certificate per thousand tokens of the application, and "
          f"the exact decomposition's localization lift at $d={tex_int(N['protocol.d'])}$ (precision over base rate, median over families).",
          "tab:rq4", star=False)


def fig_wild():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = {"proxy": "proxy_v3", "malnet": "malnet", "piggyback": "piggyback", "cic": "cic_andmal"}
    markers = {"labeled": ("o", "identifier"), "combined": ("s", "combined"), "fuzzy": ("^", "fuzzy"), "api": ("v", "api"),
               "structure": ("D", "degree")}
    colors = {"labeled": "#2a78d6", "combined": "#eb6834", "fuzzy": "#1baf7a", "api": "#eda100", "structure": "#e87ba4"}
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    seen = set()
    for yi, c in enumerate(CORPORA):
        pth = os.path.join(DATA, "cross_dataset", f"wild_colorings_{tags[c]}_gneardup_fcg.json")
        if os.path.exists(pth):
            d = json.load(open(pth))
        else:
            d = {}
        by_coloring = {}
        for k, v in d.items():
            if isinstance(v, dict) and "wild_fp" in v:
                by_coloring[k] = (v.get("wild_fp"), g(v, "abstain", "auroc"), g(v, "abstain", "wild_fp_at_tau95"))
        put(f"{c}.wild_by_coloring", by_coloring)
        for k, (m, name) in markers.items():
            v = d.get(k) or {}
            if "wild_fp" in v:
                # label each coloring once in the legend
                if name not in seen:
                    label = name
                else:
                    label = None
                ax.scatter([v["wild_fp"]], [len(CORPORA) - 1 - yi], marker=m, color=colors[k], s=26, edgecolors="white",
                           linewidths=0.5, zorder=3, label=label)
                seen.add(name)
    ticklabels = []
    for c in CORPORA:
        ticklabels.append(LABEL[c])
    ticklabels.reverse()
    ax.set_yticks(range(len(CORPORA)), ticklabels, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of Play apps assigned a non-benign class", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, frameon=False, fontsize=6.5, ncol=5, loc="lower center", bbox_to_anchor=(0.4, 1.0), handletextpad=0.2, columnspacing=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "wild_colorings.pdf"))
    plt.close(fig)


def rq3_extra():
    """Graph-level steered attack, attribution evenness vs certificate, randomized smoothing, per corpus."""
    rows = []
    idtag = {"proxy": "proxy_v3", "malnet": "malnet", "piggyback": "piggyback", "cic": "cic_andmal"}
    for c in CORPORA:
        sg = load(TAG[c], "tier4", "p3", "steer_graph.json") or {}
        ev = load(TAG[c], "tier4", "e4d_evenness.json") or {}
        sm = load(idtag[c], "exp_tier3_certificates.json") or {}
        steer = {}
        for k in ("n", "k_graph_median", "never_flipped_frac"):
            steer[k] = sg.get(k)
        put(f"{c}.steer_graph", steer)
        per_family = {}
        for f, v in (ev.get("per_family") or {}).items():
            per_family[f] = g(v, "e1_vs_kcert", "spearman")
        put(f"{c}.evenness", {"n": ev.get("n"), "e1_spearman": g(ev, "e1_vs_kcert", "spearman"), "e1_p": g(ev, "e1_vs_kcert", "spearman_p"),
                              "per_family": per_family})
        put(f"{c}.smoothing", {"apps": sm.get("apps"), "abstain": g(sm, "smoothed_pA", "abstain_fraction"),
                               "pA_median": g(sm, "smoothed_pA", "median"), "passes": g(sm, "smoothed_pA", "cost_forward_passes")})
        rows.append([LABEL[c], f"{sg.get('n', 0)}", f"{sg.get('never_flipped_frac', 0):.2f}",
                     f"{sg.get('k_graph_median', 0):.0f}", f"{g(ev, 'e1_vs_kcert', 'spearman'):+.2f}",
                     f"{g(sm, 'smoothed_pA', 'abstain_fraction'):.3f}", f"{sm.get('apps', 0)}"])
    table("rq3_extra",
          ["\\textbf{Corpus}", "\\textbf{Attacked}", "\\textbf{Never flipped}", "\\textbf{Stubs}",
           "\\textbf{Evenness $\\rho$}", "\\textbf{Smoothing abstains}", "\\textbf{Smoothed apps}"],
          rows,
          "Problem-space attack, attribution evenness and randomized smoothing on the four corpora. Attacked: staged "
          f"applications attacked by the graph-level steered attack; Never flipped: share not flipped within {N['protocol.stub_budget']:,} "
          "synthesized stubs; Stubs: median stubs when flipped. Evenness $\\rho$: Spearman correlation of attribution "
          "evenness (E1) with the first-order certificate. Smoothing: share of correctly classified applications on "
          f"which smoothing over node deletion ($\\beta={N['protocol.smoothing']['beta']}$, $N={N['protocol.smoothing']['samples']}$) abstains.",
          "tab:rq3extra", star=False)


def extras():
    idtag = {"proxy": "proxy_v3", "malnet": "malnet", "piggyback": "piggyback", "cic": "cic_andmal"}
    for c in CORPORA:
        td = load(TAG[c], "tier4", "v4", "temporal_drift.json") or {}
        w = g(td, "fields", "first_seen", "within_class") or {}
        temporal = {}
        for k in ("f1_temporal", "f1_random", "gap"):
            temporal[k] = w.get(k)
        put(f"{c}.temporal", temporal)
        ge = load(idtag[c], "exp_xai_gnnexplainer.json") or {}
        pr = ge.get("precision") or {}
        if pr.get("base"):
            put(f"{c}.gnnexplainer", {"exact_ratio": pr["exact"] / pr["base"], "gnnexplainer_ratio": pr["gnnexplainer"] / pr["base"],
                                      "degree_ratio": pr["degree"] / pr["base"], "graphs": ge.get("graphs")})
        if c == "proxy":
            lofo_tag = "proxy_v3"
        else:
            lofo_tag = idtag[c]
        ts = load(lofo_tag, "tier4", "v4", "openset_lofo.json") or {}
        put(f"{c}.openset_dex_msp", g(ts, "novelty", "msp", "wild"))


def fig_ladder():
    """Linear vs best flexible macro-F1 per corpus and coloring (one plot, absolute 0-1 axis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    lin = []
    best = []
    sig = []
    for c in CORPORA:
        lad = load(TAG[c], "tier4", "l1_ladder_summary.json") or {}
        for rep, key, _ in COLORINGS:
            r = lad.get(rep)
            if not r:
                continue
            flex = {}
            for m, v in r.items():
                if m != "linear":
                    flex[m] = v
            # best flexible model by F1 (first one on ties)
            bm, bv = max(flex.items(), key=lambda kv: kv[1].get("f1", -1))
            rows.append(f"{LABEL[c]}, {key}")
            lin.append(r["linear"]["f1"])
            best.append(bv["f1"])
            sig.append(bv.get("p", 1) < 0.05 and bv["f1"] > r["linear"]["f1"])
    # rows top to bottom
    y = []
    for i in range(len(rows) - 1, -1, -1):
        y.append(i)
    fig, ax = plt.subplots(figsize=(3.4, 4.6))
    for yy, a, b in zip(y, lin, best):
        ax.plot([a, b], [yy, yy], color="#c3c2b7", lw=1, zorder=1)
    ax.scatter(lin, y, color="#2a78d6", s=18, zorder=2, label="tuned linear")
    facecolors = []
    for s_ in sig:
        if s_:
            facecolors.append("#eb6834")
        else:
            facecolors.append("white")
    ax.scatter(best, y, facecolors=facecolors, edgecolors="#eb6834", s=18,
               zorder=3, label="best flexible (filled: significantly better)")
    ax.set_yticks(y, rows, fontsize=6.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("macro-F1", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=1, frameon=False, fontsize=6.5)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "ladder_corpora.pdf"))
    plt.close(fig)


ALL_STEPS = ["rq1", "rq2", "rq3", "rq3_extra", "rq_open", "rq_cfg", "rq4", "extras", "tab_inflation",
             "fig_ladder", "fig_localization", "fig_frontier", "fig_wild"]


def main():
    steps = sys.argv[1:] or ALL_STEPS
    os.makedirs(OUT, exist_ok=True)
    # the captions need the protocol parameters whatever steps run
    protocol()
    # run the requested steps by name (the CLI takes step names), then write numbers.json
    for s in steps:
        globals()[s]()
    json.dump(dict(N), open(NUM, "w"), indent=1, sort_keys=True, default=str)
    print(f"{len(N)} numbers -> {os.path.relpath(NUM, ART)}")


if __name__ == "__main__":
    main()
