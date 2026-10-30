# Capstone project report

Chandigarh University UG project report for *An Interpretable Multi-Modal Spam Detection
Framework Using Explainable AI and Machine Learning*.

| Path | Contents |
|---|---|
| `report.tex` | Report source (pdfLaTeX) |
| `report.pdf` | Compiled report |
| `figures/` | All figures referenced by the report |
| `scripts/make_shap_figures.py` | Regenerates the three SHAP figures with exact TreeSHAP on the sparse features the model scores |
| `templates/cu_format_sample_report.tex` | Sample report in the university format (reference only; not part of this report) |

## Build

```bash
latexmk -pdf report.tex        # or run pdflatex two or three times
```
On Overleaf, upload this folder with its `figures/` sub-folder and set `report.tex` as the main document.

## Regenerate the SHAP figures (from the repository root)

```bash
python -m src.train                          # creates models/{sms,email}_detector.joblib
python report/scripts/make_shap_figures.py
```
