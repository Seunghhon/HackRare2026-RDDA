"""
RareDx — FastAPI backend
========================
Serves:
  GET  /api/symptoms?q=<query>   → autocomplete (HPO id + name)
  POST /api/diagnose              → diagnosis + next-symptom recommendations
  GET  /                          → serves rare_disease_ui (2).html  (main UI)
  GET  /v2                        → serves rare_disease_ui2.html     (previous UI)
  GET  /v1                        → serves rare_disease_ui.html      (legacy UI)
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load model & metadata ──────────────────────────────────────────────────
print("Loading model...")
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "metadata.pkl"), "rb") as f:
    meta = pickle.load(f)

symptom_list = meta["symptom_list"]
extra_cols   = meta.get("extra_cols", [])
id_to_name   = meta["id_to_name"]
name_to_id   = meta["name_to_id"]
s_idx        = meta["s_idx"]
extra_idx    = meta.get("extra_idx", {})
le           = meta["le"]
disease_list = list(le.classes_)
d_idx        = {d: i for i, d in enumerate(disease_list)}
n_features   = len(symptom_list) + len(extra_cols)

# ── Load gene data ─────────────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "data")
genes_df = pd.read_csv(os.path.join(DATA_DIR, "rare_diseases_genes.csv"))
genes_df["disease_id"] = "ORPHA:" + genes_df["OrphaCode"].astype(str)

def _priority(t):
    t = str(t)
    if "Disease-causing" in t:      return 1
    if "Major susceptibility" in t: return 2
    if "Candidate" in t:            return 3
    if "Biomarker" in t:            return 4
    return 5

def _test_label(t):
    t = str(t)
    if "loss of function"          in t: return "Loss-of-function variant testing"
    if "gain of function"          in t: return "Gain-of-function variant testing"
    if "somatic"                   in t: return "Somatic variant testing"
    if "Disease-causing germline"  in t: return "Germline variant testing"
    if "Major susceptibility"      in t: return "Susceptibility gene testing"
    if "Biomarker"                 in t: return "Biomarker testing"
    if "Candidate"                 in t: return "Candidate gene testing (ref)"
    return "Genetic testing"

genes_df["priority"]   = genes_df["AssociationType"].apply(_priority)
genes_df["test_label"] = genes_df["AssociationType"].apply(_test_label)

# ── Build disease-symptom matrix for information gain ─────────────────────
print("Building disease-symptom matrix...")
X_matrix = np.zeros((len(disease_list), len(symptom_list)), dtype=np.float32)
hpo_path = os.path.join(BASE_DIR, "phenotype_to_genes.txt")
if os.path.exists(hpo_path):
    hpo_raw = pd.read_csv(hpo_path, sep="\t")
    hpo = (hpo_raw[hpo_raw["disease_id"].str.startswith("ORPHA")]
           [["hpo_id", "disease_id"]].drop_duplicates())
    for row in hpo.itertuples(index=False):
        di = d_idx.get(row.disease_id)
        si = s_idx.get(row.hpo_id)
        if di is not None and si is not None:
            X_matrix[di, si] = 1.0
    print("Matrix ready.")
else:
    print("WARNING: phenotype_to_genes.txt not found — next-symptom recs disabled.")

print(f"Ready — {len(disease_list)} diseases / {len(symptom_list)} symptoms\n")

# ── Helpers ────────────────────────────────────────────────────────────────
def safe_proba(vec: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = model.decision_function(vec)[0]
    raw -= raw.max()
    exp = np.exp(np.clip(raw, -500, 500))
    return exp / exp.sum()

def entropy(p: np.ndarray) -> float:
    p = p[p > 1e-10]
    return float(-np.sum(p * np.log2(p)))

def recommend_next(vec, proba, already_ids, top_k=5, candidate_n=60):
    cur_h    = entropy(proba)
    top10    = np.argsort(proba)[::-1][:10]
    sym_freq = X_matrix[top10].sum(axis=0)
    excluded = set(already_ids)

    candidates = [
        symptom_list[i]
        for i in np.argsort(sym_freq)[::-1]
        if symptom_list[i] not in excluded and sym_freq[i] > 0
    ][:candidate_n]

    results = []
    top1 = int(np.argmax(proba))
    prob_now = float(proba[top1]) * 100

    for hid in candidates:
        si = s_idx[hid]
        v_yes = vec.copy(); v_yes[0, si] = 1.0
        v_no  = vec.copy(); v_no[0, si]  = 0.0
        p_yes_arr = safe_proba(v_yes)
        p_no_arr  = safe_proba(v_no)

        p_yes_prior = float(sym_freq[si]) / len(top10)
        p_no_prior  = 1.0 - p_yes_prior
        ig = cur_h - (p_yes_prior * entropy(p_yes_arr) + p_no_prior * entropy(p_no_arr))
        if ig <= 0:
            continue

        # Top-3 diseases that change most when symptom is present
        yes_deltas = sorted(
            [
                {
                    "code": le.classes_[i],
                    "name": genes_df[genes_df["disease_id"] == le.classes_[i]]["DiseaseName"].values[0]
                             if not genes_df[genes_df["disease_id"] == le.classes_[i]].empty
                             else le.classes_[i],
                    "now":  round(float(proba[i]) * 100, 2),
                    "after": round(float(p_yes_arr[i]) * 100, 2),
                    "delta": round((float(p_yes_arr[i]) - float(proba[i])) * 100, 2),
                }
                for i in range(len(proba))
                if abs(float(p_yes_arr[i]) - float(proba[i])) * 100 > 0.01
            ],
            key=lambda x: -x["delta"]
        )[:3]

        no_deltas = sorted(
            [
                {
                    "code": le.classes_[i],
                    "name": genes_df[genes_df["disease_id"] == le.classes_[i]]["DiseaseName"].values[0]
                             if not genes_df[genes_df["disease_id"] == le.classes_[i]].empty
                             else le.classes_[i],
                    "now":  round(float(proba[i]) * 100, 2),
                    "after": round(float(p_no_arr[i]) * 100, 2),
                    "delta": round((float(p_no_arr[i]) - float(proba[i])) * 100, 2),
                }
                for i in range(len(proba))
                if abs(float(p_no_arr[i]) - float(proba[i])) * 100 > 0.01
            ],
            key=lambda x: -x["delta"]
        )[:3]

        results.append({
            "id":        hid,
            "n":         id_to_name.get(hid, hid),
            "ig":        round(ig, 4),
            "stars":     min(5, max(1, int(ig * 10))),
            "now":       round(prob_now, 3),
            "yes":       round(float(p_yes_arr[top1]) * 100, 3),
            "no":        round(float(p_no_arr[top1])  * 100, 3),
            "yes_top":   yes_deltas,
            "no_top":    no_deltas,
        })

    results.sort(key=lambda x: -x["ig"])
    return results[:top_k]

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="RareDx API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve the UI ───────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "rare_disease_ui (2).html"))

@app.get("/v2", include_in_schema=False)
def serve_ui_v2():
    return FileResponse(os.path.join(BASE_DIR, "rare_disease_ui2.html"))

@app.get("/v1", include_in_schema=False)
def serve_ui_v1():
    return FileResponse(os.path.join(BASE_DIR, "rare_disease_ui.html"))

# ── Symptom autocomplete ───────────────────────────────────────────────────
@app.get("/api/symptoms")
def search_symptoms(q: str = "", limit: int = 10):
    q = q.strip().lower()
    if len(q) < 2:
        return []
    results = sorted(
        [{"id": hid, "n": name}
         for name, hid in name_to_id.items()
         if q in name.lower() and hid in s_idx],
        key=lambda x: len(x["n"])
    )
    return results[:limit]

# ── Diagnose ───────────────────────────────────────────────────────────────
class DiagnoseRequest(BaseModel):
    symptom_ids: List[str]              # list of HPO IDs e.g. ["HP:0001250"]
    onset_key:   Optional[str] = None   # e.g. "onset_Adult"
    inh_key:     Optional[str] = None   # e.g. "inh_AR"
    prior_tests: Optional[List[str]] = []  # e.g. ["prior_panel", "prior_wes"]
    top_n:       int = 5

@app.post("/api/diagnose")
def diagnose(req: DiagnoseRequest):
    # Validate IDs
    valid = [hid for hid in req.symptom_ids if hid in s_idx]
    if not valid:
        return JSONResponse({"error": "No valid HPO IDs found."}, status_code=400)

    # Build feature vector
    vec = np.zeros((1, n_features), dtype=np.float64)
    for hid in valid:
        vec[0, s_idx[hid]] = 1.0
    offset = len(symptom_list)
    for key in [req.onset_key, req.inh_key]:
        if key and key in extra_idx:
            vec[0, offset + extra_idx[key]] = 1.0

    proba   = safe_proba(vec)
    top_idx = np.argsort(proba)[::-1][:req.top_n]

    diseases = []
    for rank, idx in enumerate(top_idx, 1):
        did  = le.classes_[idx]
        code = did  # e.g. "ORPHA:101"
        prob = round(float(proba[idx]) * 100, 3)

        # Disease name from genes_df
        name_rows = genes_df[genes_df["disease_id"] == did]["DiseaseName"]
        dname = name_rows.values[0] if not name_rows.empty else did

        # Genes
        gene_rows = genes_df[
            (genes_df["disease_id"] == did) &
            (genes_df["AssociationStatus"] == "Assessed")
        ].sort_values("priority")

        genes = [
            {
                "s": row["GeneSymbol"],
                "l": row["test_label"],
                "f": row["GeneName"],
                "p": int(row["priority"]),
            }
            for _, row in gene_rows.iterrows()
        ]

        diseases.append({
            "r":     rank,
            "code":  code,
            "name":  dname,
            "prob":  prob,
            "genes": genes,
        })

    # Next-symptom recommendations
    already = set(valid)
    next_syms = recommend_next(vec, proba, already)

    return {
        "diseases": diseases,
        "next":     next_syms,
        "entropy":  round(entropy(proba), 2),
    }

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
