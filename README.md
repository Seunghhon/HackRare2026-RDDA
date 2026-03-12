# Rare Disease Diagnostic Assistant

### Abstract
A Harvard hackathon.
We developed a symptom-driven rare disease diagnostic tool powered by a machine learning model trained on Orphanet/HPO data.  

**For research use only. Not intended for clinical diagnosis.**

---

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | Python · FastAPI · Uvicorn |
| ML Model | scikit-learn SGDClassifier (log_loss) |
| Training Data | Orphanet / HPO (`phenotype_to_genes.txt`) |
| Data Processing | pandas · NumPy |
| Frontend | Vanilla HTML/CSS/JS |

## Model

| Item | Value |
|---|---|
| Algorithm | SGDClassifier (log_loss = Logistic Regression) |
| Diseases covered | ~2,400 ORPHA-coded diseases |
| Symptoms covered | ~6,200 HPO terms |
| Features | HPO symptom binary flags + onset + inheritance |
| Epochs | 50 · L2 α = 0.001 |
| Next-symptom ranking | Shannon entropy / Information Gain |

RareDx takes HPO (Human Phenotype Ontology) symptoms as input and returns ranked candidate rare diseases with probabilities, recommended genetic tests, and the next most informative symptoms to ask about — ranked by information gain.

```
Symptoms (HPO IDs)
      │
      ▼
 SGDClassifier (Logistic Regression)
      │
      ├─▶ Top-N candidate diseases + probabilities
      ├─▶ Recommended genetic tests (from Orphanet gene data)
      └─▶ Next symptoms to confirm (by entropy / information gain)
```
