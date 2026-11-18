#!/usr/bin/env python3
"""Compact tables of the 10-page paper (regenerated/tables_10p), from regenerated/numbers.json written by
robustness_numbers.py. Run robustness_numbers.py first.
"""
import json
import os

ART = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.environ.get("ROBUSTNESS_OUT", os.path.join(ART, "regenerated"))
N = json.load(open(os.path.join(GEN, "numbers.json")))
OUT = os.path.join(GEN, "tables_10p")
C = ["proxy", "malnet", "piggyback", "cic"]
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
L = {"proxy": "Proxy", "malnet": "MalNet", "piggyback": "Piggyback", "cic": "CIC-AndMal"}


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def f2(x):
    return "--" if x is None else f"{x:.2f}"


def g(*ks):
    d = N
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def table(name, colspec, header, rows, caption, label):
    lines = ["\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table*}"]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name + ".tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    # corpora table
    rows = []
    # label and payload-truth columns: class counts and payload families from numbers.json
    meta = {"proxy": (f"{g('proxy.classes')} providers", f"SDK namespaces ({len(g('proxy.payload_families'))})"),
            "malnet": (f"{g('malnet.classes')} families", f"SDK namespaces ({len(g('malnet.payload_families'))})"),
            "piggyback": ("original/piggybacked", "rider methods"),
            "cic": (f"{g('cic.classes')} categories", "none")}
    for c in C:
        rq4 = g(f"{c}.rq4") or {}
        n_apps = g(f"{c}.n")
        if rq4.get("rho") is not None:
            rho = f"{rq4['rho']:.3f}"
        else:
            rho = "--"
        if rq4.get("min_cut") is not None:
            cut = f"{rq4['min_cut']:.0f}"
        else:
            cut = "--"
        rows.append([L[c], meta[c][0], f"{n_apps:,}", meta[c][1], rho, cut,
                     f3(g(f"{c}.ladder.identifier.linear")), f3(g(f"{c}.drebin_cert", "linear_svm", "f1"))])
    table("corpora", "lllrlrcc",
          ["\\textbf{Corpus}", "\\textbf{Labels}", "$n$", "\\textbf{Payload truth}", "$\\rho$", "\\textbf{Cut}",
           "\\textbf{WL lin.}", "\\textbf{Drebin}"], rows,
          "The four corpora; $n$ counts the applications of the closed-world task (no benign class). $\\rho$: median payload share of non-library methods; Cut: median minimum edge cut from "
          "host roots to the payload (staged payload applications). WL lin.: tuned linear model on \\textsc{identifier} "
          "FCG features; Drebin: code-only Drebin features with a linear SVM. Macro-F1 under lineage grouping.",
          "tab:corpora")

    # linearity table
    rows = []
    for c in C:
        r = [L[c]]
        for k in ("identifier", "combined", "fuzzy"):
            b = g(f"{c}.ladder.{k}.best") or {}
            lin = g(f"{c}.ladder.{k}.linear")
            if b.get("p") is not None and b["p"] < 0.05 and b["f1"] > (lin or 0):
                sig = "$^{*}$"
            else:
                sig = ""
            r.append(f"{f3(lin)} / {f3(b.get('f1'))}{sig}")
        nb = g(f"{c}.nb_vs_logistic.identifier") or [None, None]
        r += [f"{f3(nb[0])} / {f3(nb[1])}", f3(g(f"{c}.hardmargin_err.identifier")), f2(g(f"{c}.renaming.flip_rate"))]
        rows.append(r)
    table("linearity", "lcccccc",
          ["\\textbf{Corpus}", "\\textsc{identifier}", "\\textsc{combined}", "\\textsc{fuzzy}", "\\textbf{NB / log.}",
           "\\textbf{HM err.}", "\\textbf{Renamed}"], rows,
          f"Linearity (RQ1). For each coloring: tuned linear model / best of {WORDS[N['protocol.flexible_models']]} flexible models (boosting, forest, "
          "RBF SVM, MLP) on identical folds; $^{*}$ flexible significantly better ($p<0.05$). NB / log.: naive Bayes "
          "against logistic regression on exact identifier tokens. HM err.: hard-margin SVM training error. Renamed: "
          "share of \\textsc{identifier} predictions changed by per-application renaming.", "tab:linearity")

    # localization table: largest hash dimension of the sweep and smallest deletion fraction
    top = str(N["protocol.max_dim"])
    top_dim = "$2^{" + str(N["protocol.max_dim_log2"]) + "}$"
    base_dim = "$d=2^{" + str(N["protocol.base_dim_log2"]) + "}$"
    deleted = f"{100 * N['protocol.deletion_fraction']:.0f}\\%"
    flip_key = f"M3_flip_{N['protocol.deletion_fraction']}"
    rows = []
    for c in C:
        ex = g(f"{c}.loc1024.ratio_by_explainer") or {}
        if ex.get("lin_exact") is None:
            rows.append([L[c], "\\multicolumn{7}{c}{no payload ground truth}"])
            continue
        others = []
        for k, v in ex.items():
            if k != "lin_exact" and v is not None:
                others.append(v)
        follows = g(f"{c}.transplant", "A2c", "unseen", "follows_sdk")
        exact_20 = g(f"{c}.colour", f"labeled|clean|{top}", "ratio")
        fuzzy_20 = g(f"{c}.colour", f"fuzzy|clean|{top}", "ratio")
        renamed_20 = g(f"{c}.colour", f"labeled|renamed|{top}", "ratio")
        flip = g(f"{c}.mx", "lin_exact", flip_key)
        if follows is not None:
            follows_cell = f"{follows:.2f}"
        else:
            follows_cell = "--"
        rows.append([L[c], f"{ex['lin_exact']:.2f}", f"{min(others):.2f}--{max(others):.2f}",
                     f"{exact_20:.2f}", f"{fuzzy_20:.2f}", f"{renamed_20:.2f}", f"{flip:.2f}", follows_cell])
    table("localization", "lccccccc",
          ["\\textbf{Corpus}", "\\textbf{Exact}", "\\textbf{Others}", f"\\textbf{{Exact {top_dim}}}",
           f"\\textbf{{\\textsc{{fuzzy}} {top_dim}}}", f"\\textbf{{Renamed {top_dim}}}", f"\\textbf{{Del.\\ {deleted}}}",
           "\\textbf{Follows payload}"], rows,
          "Localization against payload ground truth (RQ2): precision at $|$target$|$ over the base rate, median over "
          f"payload families, at {base_dim} unless stated. Others: range over linear SHAP variants, TreeSHAP (path and "
          f"interventional, forest and boosting) and MLP gradients. Del.\\ {deleted}: share of predictions changed by "
          f"deleting the top-ranked {deleted} of methods. Follows payload: share of payload-into-foreign-host transplants "
          "whose prediction follows the payload (unseen hosts).", "tab:localization")

    # robustness table
    rows = []
    for c in C:
        fi = g(f"{c}.frontier.identifier") or {}
        fc = g(f"{c}.frontier.combined") or {}
        cc = g(f"{c}.cfg_cert") or {}
        cfg = cc.get("cfg|identifier") or {}
        dep = cc.get("concat|identifier") or {}
        sg = g(f"{c}.steer_graph") or {}
        tg = g(f"{c}.tightness") or {}
        tg = tg.get("labeled") or {}
        drebin_k = g(f"{c}.drebin_cert", "linear_svm", "k_median")
        rows.append([L[c], f"{tg['k_grad_median']:.0f}--{tg['k_steer_median']:.0f}",
                     f"{fc.get('k_steer_median', 0):.0f}",
                     f"{cfg.get('k_certgrad_median', 0):,.0f}", f"{dep.get('k_block_median', 0):,.0f}--{dep.get('k_attack_median', 0):,.0f}",
                     f2(fi.get("rand_never_frac")), f2(sg.get("never_flipped_frac")),
                     f"{drebin_k:.0f}"])
    table("robustness", "lccccccc",
          ["\\textbf{Corpus}", "\\textsc{identifier}", "\\textsc{combined} attack", "\\textbf{CFG id.}", "\\textbf{Deployed}",
           "\\textbf{Rand.\\ never}", "\\textbf{Graph never}", "\\textbf{Drebin}"], rows,
          "Certified robustness to code addition (RQ3), median added tokens over staged applications: first-order certificate "
          "(Theorem~\\ref{thm:certgrad}; block form, Theorem~\\ref{thm:certblock}, for the deployed CFG and FCG model)--greedy "
          "aimed attack; for \\textsc{combined}, the attack alone. No attack succeeded within its certificate on any corpus. Rand.\\ never: share that random padding "
          f"never flips within {N['protocol.padding_budget']:,} tokens. Graph never: share the graph-level attack with synthesized method names "
          f"never flips within {N['protocol.stub_budget']:,} stubs. Drebin: exact certificate in added features.", "tab:robustness")

    # open-world table
    rows = []
    for c in C:
        ab = g(f"{c}.open.abstain") or {}
        t = g(f"{c}.temporal") or {}
        fp_lin = g(f"{c}.open.O1", "linear", "wild_fp_rate")
        fp_rf = g(f"{c}.open.O1", "rf", "wild_fp_rate")
        if "auroc_proxy_vs_wild" in ab:
            abstain = f"{ab['auroc_proxy_vs_wild']:.2f} / {ab['wild_fp_at_tau']:.2f}"
        else:
            abstain = "n/a"
        bundle_success = g(f"{c}.bundle", "greedy", "success")
        bundle_tokens = g(f"{c}.bundle", "greedy", "median_tokens_when_successful")
        rows.append([L[c], f3(g(f"{c}.open.O2", "linear", "macro_f1")),
                     f"{fp_lin:.2f} / {fp_rf:.2f}",
                     abstain,
                     f"{bundle_success:.2f} / {bundle_tokens:,.0f}",
                     f"{t.get('f1_temporal', 0):.2f} / {t.get('f1_random', 0):.2f}"])
    table("openworld", "lccccc",
          ["\\textbf{Corpus}", "\\textbf{F1 + benign}", "\\textbf{Play FP lin.\\ / RF}", "\\textbf{Abstain AUROC / FP@95}",
           "\\textbf{Bundling succ.\\ / tokens}", "\\textbf{Later / random}"], rows,
          f"The open world and time. F1 + benign: macro-F1 with the corpus's benign class added. Play FP: share of {N['protocol.play_screened']:,} "
          "marker-screened Google Play applications assigned a non-benign class. Abstain: model without the benign "
          "class, AUROC of known against Play applications and share of Play applications above the threshold keeping "
          "95\\% of known ones. Bundling: success rate and median tokens of evasion by bundling unused benign packages. "
          "Later / random: macro-F1 on each class's later half by first-scan date against grouped random splits.",
          "tab:openworld")
    print("tables ->", os.path.relpath(OUT, ART))
