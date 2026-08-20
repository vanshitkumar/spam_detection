"""End-to-end smoke test: pipeline, XAI layer and the Streamlit demo app.

Run with:  python tests/smoke_test.py   (after `python -m src.train`)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPAM_SMS = ("URGENT! You have WON a £2,000 cash prize GUARANTEED. "
            "Call 09061701461 now or visit www.claim-your-prize.com to claim!")
HAM_SMS = "Hey, are we still meeting for lunch at 1pm? I might be 10 min late."
SPAM_EMAIL = ("Subject: Final notice — verify your account now. Dear customer, your bank "
              "account has been suspended. Click http://secure-verify-login.com within 24 "
              "hours and confirm your password to avoid permanent closure. 100% free refund "
              "of $500 guaranteed!!!")
HAM_EMAIL = ("Subject: Q3 planning meeting notes. Hi team, attached are the notes from "
             "yesterday's planning session. Please review the action items assigned to "
             "you before Friday's follow-up. Thanks, Sarah")


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}  {detail}")
    return condition


def main() -> int:
    from src.explain import build_shap_explainer, lime_explain, shap_explain_text, top_shap_features
    from src.models import SpamDetector
    from src.config import MODELS_DIR

    ok = True
    print("== Detector pipeline ==")
    cases = {"sms": (SPAM_SMS, HAM_SMS), "email": (SPAM_EMAIL, HAM_EMAIL)}
    for modality, (spam_msg, ham_msg) in cases.items():
        det = SpamDetector.load(MODELS_DIR / f"{modality}_detector.joblib")
        r_spam, r_ham = det.analyse(spam_msg), det.analyse(ham_msg)
        ok &= check(f"{modality}: spam detected", r_spam["label"] == "spam",
                    f"p={r_spam['spam_probability']:.3f}")
        ok &= check(f"{modality}: ham detected", r_ham["label"] == "ham",
                    f"p={r_ham['spam_probability']:.3f}")
        ok &= check(f"{modality}: 3 base models vote", len(r_spam["per_model"]) == 3)

        t = time.time()
        lime_exp = lime_explain(det, spam_msg, num_samples=500)
        pairs = lime_exp.as_list(label=1)
        ok &= check(f"{modality}: LIME returns weights", len(pairs) > 0,
                    f"top={pairs[0][0]!r} ({time.time() - t:.1f}s)")

        t = time.time()
        shap_exp = shap_explain_text(det, build_shap_explainer(det), spam_msg)
        top = top_shap_features(shap_exp, k=3)
        ok &= check(f"{modality}: SHAP returns values", len(top) == 3,
                    f"top={top[0][0]!r} ({time.time() - t:.1f}s)")

    print("== Streamlit app (simulated session) ==")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"),
                          default_timeout=180)
    at.run()
    ok &= check("app renders without exception", not at.exception,
                str(at.exception[0].value) if at.exception else "")
    at.text_area(key="msg_text").set_value(SPAM_SMS)
    at.button[0].set_value(True)
    at.run()
    ok &= check("analyse click renders without exception", not at.exception,
                str(at.exception[0].value) if at.exception else "")
    rendered = " ".join(str(m.value) for m in at.markdown)
    ok &= check("verdict shown", "SPAM" in rendered.upper())

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
