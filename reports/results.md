# Results

## SMS

5,158 messages (12.4% spam) — 4,126 train / 1,032 test, 3,015 features.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 (5-fold) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9855 | 0.9520 | 0.9297 | 0.9407 | 0.9941 | 0.9554 ± 0.0195 |
| Random Forest | 0.9845 | 0.9516 | 0.9219 | 0.9365 | 0.9929 | 0.9558 ± 0.0169 |
| XGBoost | 0.9845 | 0.9667 | 0.9062 | 0.9355 | 0.9941 | 0.9586 ± 0.0152 |
| **Hybrid Ensemble** | 0.9855 | 0.9669 | 0.9141 | 0.9398 | 0.9941 | — |

## Email

12,000 messages (48.3% spam) — 9,600 train / 2,400 test, 5,015 features.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 (5-fold) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9833 | 0.9753 | 0.9905 | 0.9829 | 0.9979 | 0.9797 ± 0.0041 |
| Random Forest | 0.9746 | 0.9629 | 0.9853 | 0.9740 | 0.9973 | 0.9669 ± 0.0040 |
| XGBoost | 0.9750 | 0.9645 | 0.9845 | 0.9744 | 0.9969 | 0.9732 ± 0.0032 |
| **Hybrid Ensemble** | 0.9825 | 0.9705 | 0.9940 | 0.9821 | 0.9983 | — |
