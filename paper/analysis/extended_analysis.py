"""Extended statistical analysis reported in the paper.

Re-trains both specialists with the project's own pipeline (deterministic: it
reproduces reports/metrics.json exactly) and computes
  1. bootstrap 95% confidence intervals for every metric,
  2. paired significance tests (exact McNemar; DeLong for ROC-AUC; Holm),
  3. routing accuracy, end-to-end (auto-routed) and cross-channel performance,
  4. the `enron` shortcut probe (deletion from ham / insertion into spam),
  5. the global SHAP ranking on the same 800-message subsets as src/train.py
     (exact TreeSHAP on the sparse matrix the model actually scores),
  6. a deletion test of LIME faithfulness, and
  7. prediction / explanation latency.

Run from the repository root:   python paper/analysis/extended_analysis.py
Writes paper/analysis/extended_results.json; nothing else is modified.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import binomtest, norm
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.config import CV_FOLDS, RANDOM_STATE, SHAP_BACKGROUND_SIZE, TEST_SIZE
from src.explain import lime_explain
from src.features import ENGINEERED_FEATURE_NAMES, SpamFeaturizer
from src.models import HybridVotingEnsemble, SpamDetector, build_base_models

OUT = Path(__file__).resolve().parent / "extended_results.json"
SYSTEMS = ["Logistic Regression", "Random Forest", "XGBoost", "Hybrid Ensemble"]
N_BOOT = 2000
ENRON_RE = re.compile(r"(?<![a-z])enron(?![a-z])", re.I)
EXHIBIT = ("URGENT! You have WON a £2,000 cash prize GUARANTEED. "
           "Call 09061701461 now or visit www.claim-your-prize.com to claim!")


def load_router():
    """The exact detect_modality() of app.py, extracted without running Streamlit."""
    tree = ast.parse((REPO / "app.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "detect_modality")
    ns = {"re": re}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "app.py", "exec"), ns)
    return ns["detect_modality"]


# ----------------------------------------------------------------- metrics
def threshold_metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec, "f1": f1,
            "mcc": mcc, "fpr": fp / (fp + tn) if fp + tn else 0.0}


def all_metrics(y, p):
    m = threshold_metrics(y, (p >= 0.5).astype(int))
    m["roc_auc"] = roc_auc_score(y, p)
    m["pr_auc"] = average_precision_score(y, p)
    m["brier"] = float(np.mean((p - y) ** 2))
    return m


def bootstrap_ci(y, probs, rng):
    n = len(y)
    dist = {s: [] for s in probs}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for s, p in probs.items():
            dist[s].append(all_metrics(y[idx], p[idx]))
    return {s: {k: [float(np.percentile([d[k] for d in v], 2.5)),
                    float(np.percentile([d[k] for d in v], 97.5))] for k in v[0]}
            for s, v in dist.items()}


# --------------------------------------------------------- DeLong (fast)
def _midrank(x):
    order = np.argsort(x); z = x[order]; n = len(x)
    t = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n); out[order] = t
    return out


def delong(y, p1, p2):
    order = np.argsort(-y, kind="stable")
    m = int(y.sum()); preds = np.vstack([p1, p2])[:, order]; n = preds.shape[1] - m
    tx = np.array([_midrank(r[:m]) for r in preds]); ty = np.array([_midrank(r[m:]) for r in preds])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return float(aucs[0] - aucs[1]), float(z), float(2 * norm.sf(abs(z)))


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


# ------------------------------------------------------------- training
def train(modality, transform=None):
    df = pd.read_csv(REPO / "data" / "processed" / f"{modality}.csv",
                     dtype={"label": str, "text": str}).dropna()
    texts = df["text"].tolist()
    if transform is not None:
        texts = [transform(t) for t in texts]
    y = (df["label"] == "spam").to_numpy(dtype=int)
    idx_tr, idx_te = train_test_split(np.arange(len(texts)), test_size=TEST_SIZE,
                                      stratify=y, random_state=RANDOM_STATE)
    t_tr = [texts[i] for i in idx_tr]; t_te = [texts[i] for i in idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    t0 = time.perf_counter()
    feat = SpamFeaturizer(modality=modality).fit(t_tr)
    X_tr, X_te = feat.transform(t_tr), feat.transform(t_te)
    models = build_base_models(scale_pos_weight=float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)))
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    weights = {n: float(cross_val_score(m, X_tr, y_tr, scoring="f1", cv=skf).mean())
               for n, m in models.items()}
    for m in models.values():
        m.fit(X_tr, y_tr)
    ens = HybridVotingEnsemble(models, weights)
    train_s = time.perf_counter() - t0
    det = SpamDetector(modality, feat, models, ens, metadata={})
    probs = {n: s.predict_proba(X_te)[:, 1] for n, s in {**models, "Hybrid Ensemble": ens}.items()}
    return {"det": det, "t_tr": t_tr, "t_te": t_te, "y_te": y_te, "y_tr": y_tr,
            "X_te": X_te, "probs": probs, "train_s": train_s, "weights": dict(ens.weights)}


def model_probs(det, texts):
    X = det.transform(texts)
    out = {n: m.predict_proba(X)[:, 1] for n, m in det.base_models.items()}
    out["Hybrid Ensemble"] = det.ensemble.predict_proba(X)[:, 1]
    return out


# ------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(RANDOM_STATE)
    route = load_router()
    res = {"meta": {"n_boot": N_BOOT}}
    runs = {m: train(m) for m in ("sms", "email")}

    for mod, r in runs.items():
        y, probs = r["y_te"], r["probs"]
        out = {"train_seconds": r["train_s"], "weights": r["weights"],
               "point": {s: all_metrics(y, probs[s]) for s in SYSTEMS},
               "ci": bootstrap_ci(y, probs, rng)}
        ens_ok = (probs["Hybrid Ensemble"] >= 0.5).astype(int) == y
        tests = {}
        for s in SYSTEMS[:3]:
            base_ok = (probs[s] >= 0.5).astype(int) == y
            b = int((ens_ok & ~base_ok).sum()); c = int((~ens_ok & base_ok).sum())
            p_mc = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
            d_auc, z, p_dl = delong(y, probs["Hybrid Ensemble"], probs[s])
            tests[s] = {"ens_only_correct": b, "base_only_correct": c, "mcnemar_p": float(p_mc),
                        "delta_auc": d_auc, "delong_z": z, "delong_p": p_dl}
        for key in ("mcnemar_p", "delong_p"):
            adj = holm([tests[s][key] for s in SYSTEMS[:3]])
            for s, a in zip(SYSTEMS[:3], adj):
                tests[s][key + "_holm"] = a
        out["tests"] = tests

        # error analysis
        pe = probs["Hybrid Ensemble"]
        out["false_positives"] = [{"p": float(pe[i]), "text": r["t_te"][i][:300]}
                                  for i in np.where((pe >= 0.5) & (y == 0))[0]]
        out["false_negatives"] = [{"p": float(pe[i]), "text": r["t_te"][i][:300]}
                                  for i in np.where((pe < 0.5) & (y == 1))[0]]

        # Global SHAP on the same 800-message subset as src/train.py.  XGBoost
        # treats unstored entries of a sparse matrix as *missing*, so densifying
        # the input changes the function being explained; we therefore use
        # XGBoost's native (exact) TreeSHAP on the sparse matrix the model scores.
        det = r["det"]
        bst = det.base_models["XGBoost"].get_booster()
        X_full = det.transform(r["t_te"])
        p_sparse = det.base_models["XGBoost"].predict_proba(X_full)[:, 1]
        p_dense = det.base_models["XGBoost"].predict_proba(np.asarray(X_full.todense()))[:, 1]
        sub = np.random.default_rng(RANDOM_STATE).choice(
            len(r["t_te"]), size=min(SHAP_BACKGROUND_SIZE, len(r["t_te"])), replace=False)
        X_sub = X_full[sub]
        sv = bst.predict(xgb.DMatrix(X_sub), pred_contribs=True)[:, :-1]
        imp = np.abs(sv).mean(axis=0)
        names = det.feature_names
        order = np.argsort(-imp)
        eng = set(ENGINEERED_FEATURE_NAMES)
        top = []
        for j in order[:15]:
            xj = np.asarray(X_sub[:, j].todense()).ravel()
            corr = float(np.corrcoef(xj, sv[:, j])[0, 1]) if xj.std() > 0 else 0.0
            present = xj != 0
            top.append({"feature": names[j], "mean_abs_shap": float(imp[j]), "direction_corr": corr,
                        "engineered": names[j] in eng,
                        "mean_shap_present": float(sv[present, j].mean()) if present.any() else None,
                        "mean_shap_absent": float(sv[~present, j].mean()) if (~present).any() else None})
        out["dense_vs_sparse_xgb"] = {"label_disagreements": int(((p_sparse >= .5) != (p_dense >= .5)).sum()),
                                      "n": len(p_sparse),
                                      "mean_abs_prob_diff": float(np.abs(p_sparse - p_dense).mean())}
        eng_idx = [i for i, n in enumerate(names) if n in eng]
        out["shap"] = {"top15": top,
                       "engineered_share_of_mass": float(imp[eng_idx].sum() / imp.sum()),
                       "n_engineered_in_top10": int(sum(t["engineered"] for t in top[:10])),
                       "rank_of": {w: int(np.where(order == names.index(w))[0][0]) + 1
                                   for w in ("enron", "vince", "kaminski", "hpl", "httpaddr",
                                             "phonenumbr", "numbr", "moneysymb") if w in names}}

        # latency
        single = []
        for t in r["t_te"][:200]:
            t0 = time.perf_counter(); det.predict_proba_texts([t]); single.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); det.predict_proba_texts(r["t_te"]); batch_s = time.perf_counter() - t0
        shap_t = []
        for t in r["t_te"][:100]:
            x = det.transform([t])
            t0 = time.perf_counter(); bst.predict(xgb.DMatrix(x), pred_contribs=True)
            shap_t.append(time.perf_counter() - t0)
        out["latency"] = {"single_ms_median": 1e3 * float(np.median(single)),
                          "single_ms_p95": 1e3 * float(np.percentile(single, 95)),
                          "batch_msgs_per_s": len(r["t_te"]) / batch_s,
                          "shap_ms_median": 1e3 * float(np.median(shap_t))}

        # LIME deletion test on true-positive spam messages
        tp_idx = [i for i in np.where((pe >= 0.5) & (y == 1))[0]][: (60 if mod == "sms" else 25)]
        fa = {k: {"lime": [], "random": [], "lime_flip": []} for k in (1, 3, 5)}
        lime_s = []
        for i in tp_idx:
            text = r["t_te"][i]; p0 = det.predict_proba_texts([text])[0, 1]
            t0 = time.perf_counter(); exp = lime_explain(det, text); lime_s.append(time.perf_counter() - t0)
            ids = exp.domain_mapper.indexed_string
            ranked = [fid for fid, w in sorted(exp.as_map()[1], key=lambda t: -t[1]) if w > 0]
            for k in fa:
                p1 = det.predict_proba_texts([ids.inverse_removing(ranked[:k])])[0, 1]
                fa[k]["lime"].append(p0 - p1); fa[k]["lime_flip"].append(p1 < 0.5)
                drops = []
                for _ in range(10):
                    rnd = rng.choice(ids.num_words(), size=min(k, ids.num_words()), replace=False)
                    drops.append(p0 - det.predict_proba_texts([ids.inverse_removing(list(rnd))])[0, 1])
                fa[k]["random"].append(float(np.mean(drops)))
        out["lime_faithfulness"] = {"n": len(tp_idx), "lime_s_median": float(np.median(lime_s)),
                                    **{f"k{k}": {"lime_drop": float(np.mean(v["lime"])),
                                                 "random_drop": float(np.mean(v["random"])),
                                                 "lime_flip_rate": float(np.mean(v["lime_flip"]))}
                                       for k, v in fa.items()}}
        res[mod] = out
        print(f"[{mod}] done", flush=True)

    # exhibit message (local explanation shown in the paper)
    sms_det = runs["sms"]["det"]
    a = sms_det.analyse(EXHIBIT)
    exp = lime_explain(sms_det, EXHIBIT)
    contrib = sms_det.base_models["XGBoost"].get_booster().predict(
        xgb.DMatrix(sms_det.transform([EXHIBIT])), pred_contribs=True)[0]
    o = np.argsort(-np.abs(contrib[:-1]))[:10]
    res["exhibit"] = {"analysis": a, "lime": exp.as_list(label=1),
                      "shap_base": float(contrib[-1]),
                      "shap_top10": [(sms_det.feature_names[j], float(contrib[j])) for j in o],
                      "shap_logit": float(contrib.sum())}

    # routing, end-to-end and cross-channel transfer
    s, e = runs["sms"], runs["email"]
    texts = s["t_te"] + e["t_te"]; y_all = np.concatenate([s["y_te"], e["y_te"]])
    src_lab = np.array(["sms"] * len(s["t_te"]) + ["email"] * len(e["t_te"]))
    routed = np.array([route(t) for t in texts])
    p_oracle = np.concatenate([s["probs"]["Hybrid Ensemble"], e["probs"]["Hybrid Ensemble"]])
    p_auto = np.array(p_oracle)
    for mod in ("sms", "email"):
        mask = routed == mod
        if mask.any():
            p_auto[mask] = runs[mod]["det"].predict_proba_texts([t for t, m in zip(texts, mask) if m])[:, 1]
    res["routing"] = {
        "router_acc_sms": float((routed[src_lab == "sms"] == "sms").mean()),
        "router_acc_email": float((routed[src_lab == "email"] == "email").mean()),
        "oracle": all_metrics(y_all, p_oracle), "auto": all_metrics(y_all, p_auto),
        "auto_acc_by_source": {m: float((((p_auto >= 0.5).astype(int)) == y_all)[src_lab == m].mean())
                               for m in ("sms", "email")},
        "misrouted_email_acc": float((((p_auto >= 0.5).astype(int)) == y_all)[(src_lab == "email") & (routed == "sms")].mean()),
        "cross": {"sms_model_on_email": all_metrics(e["y_te"], s["det"].predict_proba_texts(e["t_te"])[:, 1]),
                  "email_model_on_sms": all_metrics(s["y_te"], e["det"].predict_proba_texts(s["t_te"])[:, 1])}}

    # `enron` shortcut probe
    ham = [i for i in range(len(e["y_te"])) if e["y_te"][i] == 0]
    spam = [i for i in range(len(e["y_te"])) if e["y_te"][i] == 1]
    has = np.array([bool(ENRON_RE.search(t)) for t in e["t_te"]])
    has_tr = np.array([bool(ENRON_RE.search(t)) for t in e["t_tr"]])
    probe = {"share_ham_test": float(has[ham].mean()), "share_spam_test": float(has[spam].mean()),
             "share_ham_train": float(has_tr[e["y_tr"] == 0].mean()),
             "share_spam_train": float(has_tr[e["y_tr"] == 1].mean())}
    pe = e["probs"]["Hybrid Ensemble"]
    fp_idx = np.where((pe >= 0.5) & (e["y_te"] == 0))[0]
    probe["share_enron_among_fp"] = float(has[fp_idx].mean()) if len(fp_idx) else 0.0
    del_idx = [i for i in ham if has[i]]
    before = {k: v[del_idx] for k, v in e["probs"].items()}
    after = model_probs(e["det"], [ENRON_RE.sub(" ", e["t_te"][i]) for i in del_idx])
    probe["deletion"] = {"n": len(del_idx), **{k: {"mean_delta": float(np.mean(after[k] - before[k])),
                                                    "flips_to_spam": int(((before[k] < 0.5) & (after[k] >= 0.5)).sum())}
                                                for k in SYSTEMS}}
    before = {k: v[spam] for k, v in e["probs"].items()}
    after = model_probs(e["det"], [e["t_te"][i] + " enron" for i in spam])
    probe["insertion"] = {"n": len(spam), **{k: {"mean_delta": float(np.mean(after[k] - before[k])),
                                                  "flips_to_ham": int(((before[k] >= 0.5) & (after[k] < 0.5)).sum()),
                                                  "detected_before": int((before[k] >= 0.5).sum())}
                                              for k in SYSTEMS}}
    res["enron_probe"] = probe
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
