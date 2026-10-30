# Project Guide: Interpretable Multi-Modal Spam Detection

A plain-English walkthrough of the whole project, for explaining it to your supervisor.
No machine-learning background is assumed; every technical term is explained the first time
it matters, and Section 2 is a glossary you can come back to.

---

## 1. The project in one minute

**Problem.** Scam messages arrive by SMS and by e-mail. Spam filters catch most of them, but
they are usually *black boxes*: they say "spam" without saying why. Nobody can check the
decision, and hidden mistakes in the training data go unnoticed.

**What we built.** A system that

1. classifies a message as **spam** or **ham** (legitimate),
2. works for **both SMS and e-mail**, using a separate specialist model for each,
3. combines **three different machine-learning models** into one "hybrid ensemble", and
4. **explains every decision**: which words and writing features pushed the message towards
   spam.

It runs on an ordinary laptop CPU (no GPU), trains in about two minutes, and comes with an
interactive web app for demos.

**Headline results** (on messages the models never saw during training):

| Channel | Accuracy | Spam caught (recall) | Flagged messages that really were spam (precision) |
|---|---|---|---|
| SMS | 98.55% | 91.4% | 96.7% |
| E-mail | 98.25% | 99.4% | 97.0% |

**Most interesting finding.** The explanations showed that the e-mail model partly learned a
*shortcut*: the word **"enron"** appears in 44% of the legitimate e-mails in the training data
and in none of the spam. Adding that single word to a spam e-mail fools the system about a third
of the time. No accuracy number reveals this; only the explanations did.

---

## 2. Glossary: the terms you need

### Basics

| Term | Plain meaning |
|---|---|
| **Spam / ham** | Spam = unwanted or fraudulent message. Ham = legitimate message. |
| **Phishing / smishing** | Scam messages that trick people into giving passwords or money. Smishing = phishing by **SMS**. |
| **Machine learning (ML)** | Instead of writing rules by hand, we show the computer thousands of labelled examples and it learns the patterns itself. |
| **Training set / test set** | We split the data 80/20. The model learns from the 80% (training) and is graded on the 20% it has never seen (test), like an exam with unseen questions. |
| **Features** | The numbers the model actually looks at. Text has to be turned into numbers first (see TF-IDF and stylometric features). |
| **Class imbalance** | Only 12% of the SMS messages are spam. A lazy model could say "ham" every time and still be 88% correct, so we tell the models to pay extra attention to the rare spam class ("class weights"). |
| **Modality / channel** | In this project: SMS or e-mail. Both are text, but they look very different (SMS: 12 words on average; e-mail: 155 words). |

### Turning text into numbers

| Term | Plain meaning |
|---|---|
| **Normalisation** | Cleaning the text before counting words. We lower-case everything and replace links with the word `httpaddr`, phone numbers with `phonenumbr`, money amounts with `moneysymb`, other numbers with `numbr`. We *replace* rather than delete them because links and phone numbers are strong scam signals, and the placeholder words can then appear in explanations. |
| **TF-IDF** | "Term frequency × inverse document frequency". A score for each word in a message: high if the word appears often *in this message* but is rare *across all messages*. Common words like "the" get low scores; telling words like "prize" get high ones. |
| **Uni-grams / bi-grams** | Single words ("claim") and pairs of neighbouring words ("claim now"). We use both. |
| **Stylometric features** | 15 numbers describing the *style* of a message rather than its words, e.g. share of digits (`DIGIT_RATIO`), share of capital letters (`UPPERCASE_RATIO`), number of "!" (`NUM_EXCLAMATIONS`), number of links and phone numbers, average word length, and hits in a list of 46 typical scam words (`SPAM_KEYWORDS`). They are written in CAPITALS so you can tell them apart from ordinary words in the plots. |

### The three models

| Model | Plain meaning |
|---|---|
| **Decision tree** | A flowchart of yes/no questions, e.g. "Does it contain a phone number? → Is more than 10% of it digits? → spam". The building block of the next two models. |
| **Logistic Regression (LR)** | The simplest model. It gives every feature a weight (positive = points to spam, negative = points to ham), adds them up, and squashes the total into a probability between 0 and 1. Fast and surprisingly strong on text. |
| **Random Forest (RF)** | 250 decision trees, each trained on a random sample of the data and features, that **vote**. Like asking 250 people with slightly different experience and taking the majority view. Combining many trees this way is called *bagging*. |
| **XGBoost (XGB)** | "eXtreme Gradient Boosting". Also builds decision trees (300 here), but **one after another**: each new tree focuses on correcting the mistakes of the trees before it. Called *boosting*. One of the most popular models for structured data. |
| **Ensemble / soft voting** | Combining several models. Each model gives a spam probability; the ensemble takes a **weighted average**. Our weights come from each model's cross-validated F1 score, so a stronger model gets a slightly louder vote. In practice the three weights came out nearly equal (about 0.33 each). |
| **Cross-validation (5-fold)** | A fair way to measure a model *during training*: split the training data into 5 parts, train on 4 and test on the 5th, rotate 5 times, average. We use it to set the ensemble weights without touching the test set. |

### Measuring performance

Think of it as a table of four outcomes (the **confusion matrix**):

| | Model says ham | Model says spam |
|---|---|---|
| **Really ham** | True negative (TN) ✓ | False positive (FP) ✗ false alarm |
| **Really spam** | False negative (FN) ✗ missed spam | True positive (TP) ✓ |

| Metric | Plain meaning |
|---|---|
| **Accuracy** | Share of all messages classified correctly. |
| **Precision** | Of the messages we flagged as spam, how many really were spam. High precision = few false alarms. |
| **Recall** | Of all the real spam, how much we caught. High recall = little spam slips through. |
| **F1 score** | One number that balances precision and recall (their harmonic mean). |
| **ROC-AUC** | If you pick one random spam and one random ham, the chance that the model gives the spam a higher spam score. 1.0 = perfect, 0.5 = coin flip. |
| **Statistical significance (p-value)** | Whether a difference between two models is real or could be luck. Below 0.05 is usually treated as real. |

### Explainability

| Term | Plain meaning |
|---|---|
| **Explainable AI (XAI)** | Methods that show *why* a model made a decision. |
| **LIME** | "Local Interpretable Model-agnostic Explanations". Takes one message, creates about 1,000 copies with random words removed, asks the full ensemble to score each copy, and works out which words changed the score most. Output: a weight per word ("09061701461 pushed towards spam the most"). Works with any model, so it explains the **whole ensemble**. |
| **SHAP** | "SHapley Additive exPlanations". Based on game theory: it splits the credit for a prediction fairly among all features, and the pieces add up exactly to the model's output. |
| **TreeSHAP** | A fast, *exact* way to compute SHAP for tree models. We use it on the **XGBoost** part of the ensemble. |
| **Waterfall plot** | SHAP for **one** message: a chart of which features pushed its score up or down, starting from the average score. |
| **Beeswarm plot** | SHAP for **many** messages at once: which features matter most overall. Each dot is one message; red = high feature value, blue = low. |
| **Shortcut learning** | When a model learns an accidental clue in the data instead of the real concept, like a student who passes by noticing that the correct answer is always the longest option. |

---

## 3. How the system works, step by step

```
 message ─► router ─► normalise text ─► features ─► 3 models ─► weighted vote ─► verdict
 (SMS or     (SMS or    (links → httpaddr,  (TF-IDF +    (LR, RF,     (≈ ⅓ each)       + spam %
  e-mail)    e-mail?)    numbers → numbr)   15 style     XGBoost)                    + explanation
                                            features)                                (LIME, SHAP)
```

1. **Data.** Two public research datasets:
   - **UCI SMS Spam Collection**: 5,158 SMS after removing duplicates, 12.4% spam.
   - **Enron-Spam**: real e-mails from the Enron company (legitimate) mixed with spam. We used
     a 12,000-message sample that keeps the original mix: 48.3% spam.
2. **Split.** 80% training, 20% test, keeping the same spam share in both ("stratified").
3. **Normalise** the text (Section 2).
4. **Features.** TF-IDF on words and word pairs (3,000 terms for SMS, 5,000 for e-mail) plus the
   15 stylometric features. That is 3,015 numbers per SMS and 5,015 per e-mail.
5. **Train** LR, RF and XGBoost on the training set, with extra weight on the spam class.
6. **Ensemble.** Weighted average of the three probabilities; above 50% means spam.
7. **Router.** In the app, a simple rule decides which specialist to use: a message is treated as
   e-mail if it is longer than 280 characters, has a line break, or contains words like
   "subject", "dear", "regards" or "unsubscribe". The user can override it.
8. **Explain.** LIME explains the full ensemble for each message; SHAP explains the XGBoost model
   for single messages (waterfall) and overall (beeswarm).
9. **App.** A Streamlit web app (Streamlit turns a Python script into a web page) with four tabs:
   Detector, Model performance, Global explanations, About.

**Why two specialists instead of one model?** SMS and e-mail are too different. In our tests the
SMS model applied to e-mails was no better than a coin flip, and the e-mail model applied to SMS
flagged 88% of legitimate texts as spam.

**Why classical models instead of a big deep-learning model like BERT?** They reach about 98%
accuracy here, train in minutes on a CPU, and can be explained exactly. Deep models need GPUs
and are much harder to explain faithfully.

---

## 4. Results to remember

**Test set, hybrid ensemble:**

| | SMS (1,032 messages) | E-mail (2,400 messages) |
|---|---|---|
| Accuracy | 98.55% | 98.25% |
| Precision | 0.967 | 0.970 |
| Recall | 0.914 | 0.994 |
| F1 | 0.940 | 0.982 |
| ROC-AUC | 0.994 | 0.998 |
| False alarms (ham marked spam) | 4 of 904 | 35 of 1,242 |
| Missed spam | 11 of 128 | 7 of 1,158 |

- The SMS model leans towards **precision** (very few false alarms: a blocked SMS could be a
  bank OTP). The e-mail model leans towards **recall** (almost no spam gets through).
- **How the ensemble compares** (statistical tests):
  - **E-mail:** it is significantly better than Random Forest and XGBoost alone (p = 0.002) but
    statistically tied with Logistic Regression.
  - **SMS:** all models are statistically tied.
- **Honest takeaway.** All models are already above 97%, so the ensemble's value is mainly
  stability, a better precision profile and robustness (see Section 5), not raw accuracy.

---

## 5. What the explanations revealed

**1. A single message, explained.**

The test message: *"URGENT! You have WON a £2,000 cash prize GUARANTEED. Call 09061701461 now or
visit www.claim-your-prize.com to claim!"*

- **Votes:** all three models say spam (LR 100%, RF 98.7%, XGBoost 100%), so the ensemble gives
  **99.6% spam**.
- **LIME's top words:** the premium-rate phone number, then "000" (from £2,000), "claim",
  "URGENT", "prize". These are reasons a non-expert can check.
- **SHAP's top features:** mainly *style*, namely the high share of digits, long words, scam
  keywords and capital letters.

**2. Style matters as much as vocabulary (SMS).**
- The five most important SMS features are all stylometric: `DIGIT_RATIO`, `AVG_WORD_LEN`,
  `SPAM_KEYWORDS`, `UPPERCASE_RATIO`, `CHAR_COUNT`.
- Together the 15 style features carry about **75%** of the model's reasoning, although they are
  only 15 of 3,015 features.
- Short scams are betrayed by their *shape* (digits, capitals, "!") as much as by their words.

**3. The "enron" shortcut (e-mail).**
- **The clue:** "enron" is the most important e-mail feature. It is in 44% of legitimate training
  e-mails and in **zero** spam, because every legitimate e-mail in this dataset is internal
  Enron company mail. The model partly learned "mentions Enron = safe".
- **The test:** we added the single word "enron" to every spam e-mail in the test set. The
  ensemble then let **35.5%** of them through, XGBoost alone **91.7%**, Logistic Regression only
  **4.5%**.
- **Side evidence:** none of the 35 false alarms contained "enron". The legitimate e-mails the
  model got wrong were exactly the ones without the company name.
- **Lesson:** a 98%-accurate model can still have learned the wrong thing, and explanations are
  how you find out.

**4. The explanations are trustworthy (checked).** For spam messages we deleted the top 5 words
LIME pointed to. The spam score dropped about **6 times more (SMS) and 20 times more (e-mail)**
than when deleting 5 random words. So LIME really does point at the words the model relies on.

**5. A bug we found and fixed in the explanations.** The original SHAP code converted the data
to a different format ("dense") before explaining XGBoost. XGBoost reads missing values
differently in that format, so the plots explained a slightly different model; it changed the
XGBoost verdict for 729 of 1,032 SMS. The report and paper now use corrected SHAP figures. The
app and `src/explain.py` still use the old method (see Section 9).

---

## 6. Extra analysis added for the research paper

The IEEE paper (`paper/`) goes beyond the report:

- **Confidence intervals and significance tests:** how sure we can be about each number, and
  whether differences between models are real.
- **Router test:** the automatic router sends 96.4% of SMS and 85.1% of e-mails to the right
  specialist. With automatic routing, overall accuracy drops from 98.3% to 92.2%: the
  classifiers are strong, the routing rule is the weak link.
- **Cross-channel test:** each specialist fails on the other channel, which justifies having two.
- **The "enron" attack** and the **LIME deletion test** described in Section 5.
- **Speed:** about 15 ms per verdict, under 1 ms for a SHAP explanation, 0.1–0.2 s for LIME.

---

## 7. Limitations (say these before you are asked)

- The datasets are **English only and old** (2000–2012); modern WhatsApp scams, Hinglish and
  AI-written phishing are not covered.
- The e-mail accuracy is **flattered by the "enron" shortcut**; on another company's mail it
  would likely be lower.
- The **router is a simple rule** and misroutes some short e-mails (the app has a manual
  override).
- **SHAP explains only the XGBoost part**; LIME covers the full ensemble but is slower and
  slightly random.
- No deep-learning or LLM model was compared under the same tests.

---

## 8. Questions your supervisor may ask

**Why use three models instead of the best one?**
They make different kinds of mistakes (linear, bagging, boosting). On e-mail the ensemble is
significantly better than either tree model alone, and it is far harder to fool with the "enron"
trick than XGBoost alone (35.5% vs 91.7%).

**Why are the ensemble weights almost equal?**
The three models scored almost the same in cross-validation (F1 ≈ 0.96 on SMS, ≈ 0.97 on
e-mail), so their weights all came out at about 0.33.

**Why not delete links and numbers during cleaning, as many projects do?**
They are among the strongest scam signals. Replacing them with placeholder words keeps the signal
*and* lets the explanation say "the link pushed this towards spam".

**How did you handle the imbalance in SMS (only 12% spam)?**
Class weights: mistakes on spam cost more during training. We avoided generating fake training
messages (SMOTE) so every example stays a real, explainable message.

**What is the difference between LIME and SHAP here?**
LIME explains the **whole ensemble** for one message by trying ~1,000 variations: approximate,
works with anything. SHAP explains the **XGBoost model** exactly, for one message or for the
whole dataset.

**Is 98% good?**
It is in line with published results on these datasets. More importantly, we show *why* each
decision is made and found a flaw that accuracy alone hid.

**What would you do next?**
Test on newer and multilingual data (including Indian languages), remove the "enron"-type
shortcuts, replace the router with a learned model, compare against a transformer or LLM, and
test robustness against deliberately disguised spam.

---

## 9. Where things are in the repository

| Path | What it is |
|---|---|
| `app.py` | The Streamlit demo app |
| `src/config.py` | Settings: file paths, model parameters, the 46-word scam lexicon |
| `src/data_loader.py` | Downloads and cleans the two datasets |
| `src/preprocessing.py` | Text normalisation (links → `httpaddr`, etc.) |
| `src/features.py` | TF-IDF + 15 stylometric features |
| `src/models.py` | The three models and the weighted ensemble |
| `src/explain.py` | LIME and SHAP helpers (SHAP still uses the old dense method) |
| `src/train.py` | Trains everything, evaluates, draws figures |
| `tests/smoke_test.py` | 13 automatic end-to-end checks |
| `notebooks/` | Research notebook with the full analysis |
| `reports/` | Metrics, results tables and figures produced by training |
| `report/` | Capstone report (LaTeX + PDF) |
| `paper/` | IEEE research paper (LaTeX + PDF) and its analysis scripts |
| `docs/project_guide.md` | This guide |

---

## 10. Running the demo

### Step 1: one-time setup (about 5 minutes)

Open a terminal in the project folder (in VS Code: *Terminal → New Terminal*) and run:

```bash
python -m venv .venv                  # create a private Python environment
source .venv/bin/activate             # on Windows: .venv\Scripts\activate
pip install -r requirements.txt       # install the libraries (2-3 minutes)
python -m src.data_loader             # prepare the datasets (instant: they are already in data/processed)
python -m src.train                   # train both detectors (2-5 minutes); creates models/
python tests/smoke_test.py            # optional check: should end with "ALL CHECKS PASSED"
```

Tested on Python 3.12 and on Python 3.14 (the version in this project's dev container).
`python -m src.train` also rewrites the files in `reports/` with the same numbers. If
`git status` then shows them as changed, that is only new timestamps and images; you can
undo it with `git checkout reports/`.

### Step 2: start the app

```bash
streamlit run app.py
```

Open **http://localhost:8501**. In VS Code a pop-up usually offers *Open in Browser*; if not, open
the **Ports** tab and click the globe icon next to port 8501. Stop the app with `Ctrl+C`. Next
time, only `source .venv/bin/activate` and `streamlit run app.py` are needed.

### Step 3: a 5-minute demo script

Paste each message into the **Detector** tab and click **Analyse message** (leave *Auto-detect*
selected). More ready-to-paste samples are in Section 11. Expected results:

| # | Paste this | Expected |
|---|---|---|
| 1 | `URGENT! You have WON a £2,000 cash prize GUARANTEED. Call 09061701461 now or visit www.claim-your-prize.com to claim!` | **SPAM 99.6%** (SMS). All three models agree. |
| 2 | `Hey, are we still meeting for lunch at 1pm? I might be 10 min late.` | **HAM 7.9%** (SMS) |
| 3 | `Subject: Password expiry notice. Your mailbox password expires today. Click http://mail-verify-portal.com to keep your account active and avoid losing your messages. IT Support` | **SPAM 86.6%** (e-mail) |
| 4 | Message 3 with `, Enron North America` added at the very end | **HAM 42.3%**: the shortcut. XGBoost drops from 87% to 3%, Logistic Regression still says 80% spam. |
| 5 | `Subject: Q3 planning meeting notes. Hi team, attached are the notes from yesterday's planning session. Please review the action items assigned to you before Friday's follow-up. Thanks, Sarah` | **HAM 14.0%** (e-mail) |

What to point at while you talk:

1. **Message 1.** The three model votes and their weights; the message with LIME-highlighted words
   (red = pushes towards spam: the phone number is the strongest); the engineered features
   (lots of digits and capitals).
2. **Message 2.** A normal text gets a low spam score.
3. **Messages 3 → 4.** The highlight of the demo. Explain the "enron" shortcut (Section 5): the
   e-mail model learned that "Enron" means "safe", so signing a scam with the company name gets
   it through. Point out that Logistic Regression is not fooled, which is why combining models
   helps.
4. **Try an example** (dropdown): real messages from the test set.
5. **Model performance tab:** the results tables, ROC curves and confusion matrices.
6. **Global explanations tab** and **About tab:** what the models learned overall, and the
   architecture.

### Before the demo, know these

- **SHAP plots in the app use the old method.** The SHAP waterfall in the Detector tab and the
  beeswarms in the Global explanations tab still come from the "dense" method, so they can differ
  from the corrected figures in the report and paper. The LIME highlights, verdicts and
  probabilities are all correct. Fixing `src/explain.py` is a small change.
- **The router can pick the wrong channel.** An SMS containing words like "dear" or "unsubscribe"
  goes to the e-mail model. Use the **SMS / Email** buttons to override it.
- **The first analysis takes a few seconds** while the models load; after that it is quick.

### If something goes wrong

| Problem | Fix |
|---|---|
| App says *"Trained models not found"* | Run `python -m src.train` first. |
| `ModuleNotFoundError` | The environment is not active: run `source .venv/bin/activate`. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` and open that port instead. |
| Training is slow | Close other heavy programs; it normally takes 2–5 minutes, depending on the computer. |

---

## 11. Sample messages for the demo

All messages below were tested with the trained models, so the expected results are what the app shows (with *Auto-detect* selected, each goes to the right specialist). The names, links and phone numbers are invented for the demo; do not click or call them.

Good mix for a live demo: two scam SMS, two genuine SMS, two scam e-mails, one genuine e-mail, then the "Enron" trick from Section 10.

### Scam SMS

**Scam SMS 1**

```text
Congratulations! Your mobile number has won Rs 25,00,000 in our annual lucky draw. To claim your prize call 9876543210 now and share your bank details.
```
Expected: **SPAM 99.0%**

**Scam SMS 2**

```text
Your electricity connection will be disconnected tonight at 9:30 PM due to an unpaid bill. Call our officer on 9123456780 immediately to avoid disconnection.
```
Expected: **SPAM 91.6%**

**Scam SMS 3**

```text
Your parcel is on hold due to an incomplete address. Update your details within 24 hrs to avoid return: http://parcel-redelivery-help.com
```
Expected: **SPAM 78.5%**

**Scam SMS 4**

```text
ALERT: Your bank account will be blocked today. Complete your KYC now at http://kyc-verify-now.com to avoid suspension.
```
Expected: **SPAM 88.9%**

**Scam SMS 5**

```text
You have been selected for a part-time job. Earn Rs 5000 daily from home. WhatsApp 9812345670 to start today!
```
Expected: **SPAM 93.1%**

**Scam SMS 6**

```text
FREE entry! Text WIN to 80086 for a chance to win a brand new iPhone 16. T&Cs apply. 18+ only.
```
Expected: **SPAM 96.2%**

**Scam SMS 7**

```text
Your account has been credited with a cashback of Rs 2,000. Claim it before it expires today: http://claim-cashback-now.com
```
Expected: **SPAM 91.2%**

**Scam SMS 8**

```text
URGENT! You have WON a £2,000 cash prize GUARANTEED. Call 09061701461 now or visit www.claim-your-prize.com to claim!
```
Expected: **SPAM 99.6%**

### Genuine SMS

**Genuine SMS 1**

```text
Hey, are we still meeting for lunch at 1pm? I might be 10 min late.
```
Expected: **HAM 7.9%**

**Genuine SMS 2**

```text
Mom, I reached the hostel safely. Will call you after dinner.
```
Expected: **HAM 3.4%**

**Genuine SMS 3**

```text
Can you send me the notes from today's lecture? I missed the last 20 minutes.
```
Expected: **HAM 7.5%**

**Genuine SMS 4**

```text
Running late, the bus is stuck in traffic. Start the meeting without me, I'll join by 4.
```
Expected: **HAM 3.5%**

**Genuine SMS 5**

```text
Happy birthday bro! Have an awesome day, party is on me this weekend.
```
Expected: **HAM 5.4%**

**Genuine SMS 6**

```text
Ok, I'll pick up the groceries on my way home. Need anything else?
```
Expected: **HAM 1.0%**

### Scam e-mails

**Scam e-mail 1**

```text
Subject: Password expiry notice. Your mailbox password expires today. Click http://mail-verify-portal.com to keep your account active and avoid losing your messages. IT Support
```
Expected: **SPAM 86.6%**

**Scam e-mail 2**

```text
Subject: Your account has been suspended. Dear customer, we detected unusual activity. Verify your account details at http://secure-account-check.com within 24 hours or your account will be closed. Customer Service
```
Expected: **SPAM 85.2%**

**Scam e-mail 3**

```text
Subject: You have won a free gift card! Congratulations, you were selected to receive a 500 dollar gift card. Click here to claim your reward now: http://claim-gift-now.com . Offer expires today!
```
Expected: **SPAM 96.4%**

**Scam e-mail 4**

```text
Subject: Urgent: confirm your payment details. Hello, your recent order could not be processed because your payment information is outdated. Please update your billing details at http://secure-billing-update.com within 48 hours or your order will be cancelled. Thank you, Customer Support Team
```
Expected: **SPAM 81.8%**

**Scam e-mail 5**

```text
Subject: Work from home opportunity. Earn $5,000 per month working just 2 hours a day from home. No experience needed. Limited spots available, click here to register today: http://easy-income-jobs.net
```
Expected: **SPAM 95.3%**

**Scam e-mail 6**

```text
Subject: Exclusive investment offer. Dear friend, our new crypto fund offers guaranteed returns of 40% per month with zero risk. Only 50 investors will be accepted. Reply with your full name and phone number to secure your place today!
```
Expected: **SPAM 86.2%**

### Genuine e-mails

**Genuine e-mail 1**

```text
Subject: Q3 planning meeting notes. Hi team, attached are the notes from yesterday's planning session. Please review the action items assigned to you before Friday's follow-up. Thanks, Sarah
```
Expected: **HAM 14.0%**

**Genuine e-mail 2**

```text
Subject: Feedback on the draft report. Hi Vanshit, thanks for sending the draft. Overall it looks good. Please add the confusion matrices to chapter 4 and double check the references before Monday. Regards, Priya
```
Expected: **HAM 23.1%**

**Genuine e-mail 3**

```text
Subject: Team lunch on Friday. Hi everyone, we are planning a team lunch this Friday at 1 pm to celebrate the release. Let me know by Wednesday if you can join and if you have any food preferences. Cheers, Rahul
```
Expected: **HAM 7.2%**

**Genuine e-mail 4**

```text
Subject: Minutes of today's meeting. Hi all, please find the minutes of today's meeting attached. Action items: Aditi will finalise the dataset section and Naman will update the slides. Our next meeting is on Monday at 11 am. Thanks
```
Expected: **HAM 7.3%**

**Genuine e-mail 5**

```text
Subject: Lab session moved to Friday. Hi all, the machine learning lab on Thursday has been moved to Friday at 10 am in the same room. Please bring your laptops and the dataset we used last week. See you there.
```
Expected: **HAM 21.0%**

**Genuine e-mail 6**

```text
Subject: Re: library book renewal. Thanks for the reminder. I have renewed both books online and they are now due back on the 15th. Let me know if you need anything else for the project.
```
Expected: **HAM 13.6%**

### Honest misses (optional, good for discussing limitations)

These two are classified **wrongly**. Showing one shows the supervisor you understand the system's limits.

**Genuine SMS wrongly flagged**

```text
Your OTP for login is 482913. It is valid for 10 minutes. Do not share it with anyone.
```
Expected: **SPAM 52.2%** (wrong). A genuine one-time password, flagged as spam (just over the 50% line). Real OTP messages are full of digits, exactly like scams, which is why the report stresses precision on SMS: blocking an OTP is costly.

**Scam e-mail missed**

```text
Subject: Invoice payment overdue. Please find the attached invoice and make the payment of 2,450 dollars today using the link http://pay-invoice-online.net to avoid late fees. Accounts Department
```
Expected: **HAM 38.9%** (wrong). An invoice-fraud e-mail that gets through. It has no typical spam words (no "win", "free", "urgent"); Logistic Regression catches it (65% spam) but the two tree models do not. Business-style fraud is a known weakness of word-based filters.
