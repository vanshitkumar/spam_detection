"""Dataset acquisition and preparation for both modalities.

Sources (public research corpora):
* SMS   — UCI "SMS Spam Collection" (Almeida & Gómez Hidalgo, 2011), 5 574
          labelled SMS messages.
* Email — "Enron-Spam" corpus (Metsis, Androutsopoulos & Paliouras, 2006),
          ~33 000 labelled e-mails, fetched as a consolidated CSV mirror.
          Fallback: the Apache SpamAssassin public corpus.

Running ``python -m src.data_loader`` downloads whatever is missing and
writes clean ``data/processed/{sms,email}.csv`` files with columns
``label`` ('ham'/'spam') and ``text``.
"""

from __future__ import annotations

import io
import re
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .config import EMAIL_SAMPLE_SIZE, PROCESSED_DIR, RANDOM_STATE, RAW_DIR

SMS_URLS = [
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
]
ENRON_URL = "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip"
SPAMASSASSIN_BASE = "https://spamassassin.apache.org/old/publiccorpus/"
SPAMASSASSIN_FILES = [
    ("20030228_easy_ham.tar.bz2", "ham"),
    ("20030228_easy_ham_2.tar.bz2", "ham"),
    ("20030228_hard_ham.tar.bz2", "ham"),
    ("20030228_spam.tar.bz2", "spam"),
    ("20050311_spam_2.tar.bz2", "spam"),
]


def _download(url: str, dest: Path, timeout: int = 300) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research project)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        dest.write_bytes(resp.read())
    return dest


# --------------------------------------------------------------------- SMS
def _load_sms() -> pd.DataFrame:
    tsv = RAW_DIR / "sms.tsv"
    if not tsv.exists():
        try:
            _download(SMS_URLS[0], tsv)
        except Exception as exc:  # fall back to the official UCI zip
            print(f"  primary SMS mirror failed ({exc}); trying UCI…")
            zpath = _download(SMS_URLS[1], RAW_DIR / "sms_uci.zip")
            with zipfile.ZipFile(zpath) as z:
                tsv.write_bytes(z.read("SMSSpamCollection"))
    df = pd.read_csv(tsv, sep="\t", header=None, names=["label", "text"],
                     quoting=3, dtype=str, on_bad_lines="skip")
    df = df[df["label"].isin(["ham", "spam"])]
    return df


# ------------------------------------------------------------------- Email
def _clean_email_text(subject: str, body: str) -> str:
    text = f"{subject} {body}"
    text = re.sub(r"\s+", " ", text).strip()
    return text[:8000]  # very long e-mails add cost, not signal


def _load_email_enron() -> pd.DataFrame:
    zpath = _download(ENRON_URL, RAW_DIR / "enron_spam_data.zip")
    with zipfile.ZipFile(zpath) as z:
        with z.open("enron_spam_data.csv") as f:
            df = pd.read_csv(f, dtype=str)
    df = df.rename(columns={"Spam/Ham": "label"})
    subj = df["Subject"].fillna("")
    body = df["Message"].fillna("")
    df["text"] = [_clean_email_text(s, b) for s, b in zip(subj, body)]
    df["label"] = df["label"].str.strip().str.lower()
    return df[["label", "text"]]


def _load_email_spamassassin() -> pd.DataFrame:
    rows = []
    for fname, label in SPAMASSASSIN_FILES:
        path = _download(SPAMASSASSIN_BASE + fname, RAW_DIR / fname)
        with tarfile.open(fileobj=io.BytesIO(path.read_bytes()), mode="r:bz2") as tar:
            for member in tar.getmembers():
                if not member.isfile() or member.name.endswith("cmds"):
                    continue
                raw = tar.extractfile(member).read().decode("latin-1", errors="replace")
                headers, _, body = raw.partition("\n\n")
                m = re.search(r"^Subject:\s*(.*)$", headers, flags=re.M)
                subject = m.group(1) if m else ""
                body = re.sub(r"<[^>]{1,200}>", " ", body)  # strip HTML tags
                rows.append({"label": label, "text": _clean_email_text(subject, body)})
    return pd.DataFrame(rows)


def _load_email() -> pd.DataFrame:
    try:
        return _load_email_enron()
    except Exception as exc:
        print(f"  Enron mirror failed ({exc}); falling back to SpamAssassin corpus…")
        return _load_email_spamassassin()


# ----------------------------------------------------------------- shared
def _finalise(df: pd.DataFrame, modality: str, sample_size: int | None) -> pd.DataFrame:
    df = df.dropna(subset=["label", "text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 3]
    df = df[df["label"].isin(["ham", "spam"])]
    df = df.drop_duplicates(subset="text")
    if sample_size is not None and len(df) > sample_size:
        # stratified subsample that preserves the original class ratio
        df = (
            df.groupby("label", group_keys=False)[["label", "text"]]
            .apply(lambda g: g.sample(round(len(g) / len(df) * sample_size),
                                      random_state=RANDOM_STATE))
        )
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    spam_share = (df["label"] == "spam").mean()
    print(f"  {modality}: {len(df):>6,} messages | spam share {spam_share:.1%}")
    return df


def ensure_data(force: bool = False) -> dict[str, Path]:
    """Make sure processed CSVs exist; download/build them if needed."""
    paths = {m: PROCESSED_DIR / f"{m}.csv" for m in ("sms", "email")}
    if not force and all(p.exists() for p in paths.values()):
        return paths

    print("Preparing SMS dataset (UCI SMS Spam Collection)…")
    sms = _finalise(_load_sms(), "sms", sample_size=None)
    sms.to_csv(paths["sms"], index=False)

    print("Preparing Email dataset (Enron-Spam)…")
    email = _finalise(_load_email(), "email", sample_size=EMAIL_SAMPLE_SIZE)
    email.to_csv(paths["email"], index=False)

    return paths


def load_processed(modality: str) -> pd.DataFrame:
    path = ensure_data()[modality]
    return pd.read_csv(path, dtype={"label": str, "text": str}).dropna()


if __name__ == "__main__":
    import sys
    ensure_data(force="--force" in sys.argv)
    print("Done. Processed files in", PROCESSED_DIR)
