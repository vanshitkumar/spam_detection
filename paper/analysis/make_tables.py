"""Render the paper's result tables from extended_results.json.

Run from the repository root:  python paper/analysis/make_tables.py
Writes paper/tables/*.tex (table bodies \\input by main.tex).
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "extended_results.json").read_text())
OUT = HERE.parent / "tables"
OUT.mkdir(exist_ok=True)
SYS = [("Logistic Regression", "LR"), ("Random Forest", "RF"), ("XGBoost", "XGB"),
       ("Hybrid Ensemble", "Ensemble")]
MOD = [("sms", "SMS"), ("email", "E-mail")]


def f4(x):
    return f"{x:.4f}"


def ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]".replace("0.", ".")


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# main results with bootstrap CIs
rows = []
for mk, mname in MOD:
    first = True
    for s, short in SYS:
        pt, c = R[mk]["point"][s], R[mk]["ci"][s]
        cells = []
        for k in ("accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc", "brier"):
            val = f4(pt[k])
            if s == "Hybrid Ensemble":
                val = r"\textbf{" + val + "}"
            cells.append(r"\shortstack{" + val + r"\\{\tiny " + ci(*c[k]) + "}}")
        label = (r"\multirow{4}{*}{" + mname + "}") if first else ""
        first = False
        name = r"\textbf{Ensemble}" if short == "Ensemble" else short
        rows.append(f"{label} & {name} & " + " & ".join(cells) + r" \\")
    rows.append(r"\midrule" if mk == "sms" else "")
(OUT / "results.tex").write_text("\n".join(r for r in rows if r) + "\n")

# paired significance tests
rows = []
for mk, mname in MOD:
    first = True
    for s, short in SYS[:3]:
        t = R[mk]["tests"][s]
        label = (r"\multirow{3}{*}{" + mname + "}") if first else ""
        first = False
        rows.append(f"{label} & Ens.\\ vs.\\ {short} & {t['ens_only_correct']}/{t['base_only_correct']} & "
                    f"{pval(t['mcnemar_p_holm'])} & {t['delta_auc']:+.4f} & {t['delong_z']:.2f} & "
                    f"{pval(t['delong_p_holm'])} \\\\")
    if mk == "sms":
        rows.append(r"\midrule")
(OUT / "significance.tex").write_text("\n".join(rows) + "\n")

# routing and cross-channel transfer
ro = R["routing"]
lines = [("Oracle routing (pooled)", ro["oracle"]), ("Automatic routing (pooled)", ro["auto"]),
         ("SMS model $\\rightarrow$ e-mail test", ro["cross"]["sms_model_on_email"]),
         ("E-mail model $\\rightarrow$ SMS test", ro["cross"]["email_model_on_sms"])]
rows = []
for i, (name, m) in enumerate(lines):
    rows.append(f"{name} & {m['accuracy']:.4f} & {m['f1']:.4f} & {m['mcc']:.4f} & {m['roc_auc']:.4f} & "
                f"{m['fpr']:.4f} \\\\")
    if i == 1:
        rows.append(r"\midrule")
(OUT / "routing.tex").write_text("\n".join(rows) + "\n")

# enron shortcut probe
p = R["enron_probe"]
rows = []
for s, short in SYS:
    ins, de = p["insertion"][s], p["deletion"][s]
    name = r"\textbf{Ensemble}" if short == "Ensemble" else short
    rate = 100 * ins["flips_to_ham"] / ins["detected_before"]
    rows.append(f"{name} & {ins['flips_to_ham']}/{ins['detected_before']} ({rate:.1f}\\%) & "
                f"{ins['mean_delta']:+.3f} & {de['flips_to_spam']}/{p['deletion']['n']} & {de['mean_delta']:+.3f} \\\\")
(OUT / "shortcut.tex").write_text("\n".join(rows) + "\n")

# LIME deletion test + cost
rows = []
for mk, mname in MOD:
    lf = R[mk]["lime_faithfulness"]
    cells = []
    for k in ("k1", "k3", "k5"):
        cells.append(f"{lf[k]['lime_drop']:.3f} / {lf[k]['random_drop']:.3f}")
    rows.append(f"{mname} ($n={lf['n']}$) & " + " & ".join(cells) +
                f" & {100 * lf['k5']['lime_flip_rate']:.0f}\\% \\\\")
(OUT / "lime.tex").write_text("\n".join(rows) + "\n")

rows = []
for mk, mname in MOD:
    lat, lf = R[mk]["latency"], R[mk]["lime_faithfulness"]
    rows.append(f"{mname} & {R[mk]['train_seconds']:.1f} & {lat['single_ms_median']:.1f} & "
                f"{lat['batch_msgs_per_s']:,.0f} & {lat['shap_ms_median']:.2f} & {1e3 * lf['lime_s_median']:.0f} \\\\")
(OUT / "cost.tex").write_text("\n".join(rows).replace(",", "{,}") + "\n")
print("tables written to", OUT)
