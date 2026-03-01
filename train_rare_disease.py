import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("  Rare Disease Diagnostic Model Training")
print("=" * 60)

# STEP 1. Load data
print("\n[STEP 1] Loading data...")
hpo_raw = pd.read_csv(os.path.join(BASE_DIR, 'phenotype_to_genes.txt'), sep='\t')
hpo = (
    hpo_raw[hpo_raw['disease_id'].str.startswith('ORPHA')]
    [['hpo_id', 'hpo_name', 'disease_id']]
    .drop_duplicates()
)
print(f"Raw disease count : {hpo['disease_id'].nunique():,}")
print(f"Raw HPO symptom count : {hpo['hpo_id'].nunique():,}")

# STEP 2. Filtering
print("\n[STEP 2] Filtering...")

sc = hpo.groupby('hpo_id')['disease_id'].nunique()
# hpo = hpo[hpo['hpo_id'].isin(sc[(sc >= 2) & (sc <= 500)].index)]

hpo = hpo[hpo['hpo_id'].isin(sc[(sc >= 2) & (sc <= 800)].index)]

dc = hpo.groupby('disease_id')['hpo_id'].nunique()
hpo = hpo[hpo['disease_id'].isin(dc[dc >= 8].index)]

print(f"  Disease count after filtering : {hpo['disease_id'].nunique():,}")
print(f"  Symptom count after filtering : {hpo['hpo_id'].nunique():,}")

# STEP 3. Build disease x symptom matrix
print("\n[STEP 3] Building disease-symptom matrix...")

disease_list = sorted(hpo['disease_id'].unique())
symptom_list = sorted(hpo['hpo_id'].unique())
d_idx = {d: i for i, d in enumerate(disease_list)}
s_idx = {s: i for i, s in enumerate(symptom_list)}

X = np.zeros((len(disease_list), len(symptom_list)), dtype=np.float32)
for row in tqdm(hpo.itertuples(index=False), total=len(hpo), desc="  Building matrix", ncols=60):
    X[d_idx[row.disease_id], s_idx[row.hpo_id]] = 1.0

print(f"  Matrix shape : {X.shape[0]} x {X.shape[1]}")
print(f"  Memory       : {X.nbytes / 1024**2:.1f} MB")

id_to_name = hpo.drop_duplicates('hpo_id').set_index('hpo_id')['hpo_name'].to_dict()
name_to_id = {v: k for k, v in id_to_name.items()}

# STEP 4. Train/test split
le = LabelEncoder()
y = le.fit_transform(disease_list)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 5. Logistic regression training via SGD
EPOCHS = 50

print(f"\n[STEP 4] Training logistic regression... (total {EPOCHS} epochs)")
print(f"  Train samples : {len(X_train)} diseases")
print(f"  Test samples  : {len(X_test)} diseases")
print(f"  Features      : {X.shape[1]} symptoms")
print(f"  Classes       : {len(le.classes_)} diseases\n")

model = SGDClassifier(
    loss='log_loss',
    alpha=0.001,
    max_iter=1,
    warm_start=True,
    random_state=42,
    n_jobs=-1,
    learning_rate='constant',  # fixed learning rate — prevents auto decay per epoch
    eta0=0.01,                 # learning rate value
)

classes = np.arange(len(le.classes_))
best_acc = 0.0

# Weight tracking setup
# Specific  (few diseases)  → weight expected to converge positive
# Common    (many diseases) → weight expected to converge negative
LEIGH = 'ORPHA:506'
TRACK_SYMPTOMS = {
    'HP:0003128': ('Lactic acidosis', 44,  'Specific (44 diseases,  1.8%)'),
    'HP:0001250': ('Seizure        ', 752, 'Common   (752 diseases, 30.8%)'),
}

leigh_row = list(le.classes_).index(LEIGH) if LEIGH in le.classes_ else None

print(f"  Tracking target: {LEIGH} (Leigh syndrome)")
print(f"  {'Symptom':<22}  {'HPO ID':<14}  {'Diseases':>9}  Specificity")
print(f"  {'─'*65}")
for hid, (name, cnt, spec) in TRACK_SYMPTOMS.items():
    status = '✓' if hid in s_idx else '✗ filtered out'
    print(f"  {name}  {hid}  {cnt:>6}      {spec}  {status}")
print()

pbar = tqdm(range(1, EPOCHS + 1), desc="  Training", ncols=70, unit="epoch")
for epoch in pbar:
    perm = np.random.permutation(len(X_train))
    model.partial_fit(X_train[perm], y_train[perm], classes=classes)

    # Measure accuracy every 10 epochs
    if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
        acc = accuracy_score(y_test, model.predict(X_test))
        best_acc = max(best_acc, acc)
        pbar.set_postfix({
            'test_acc': f"{acc*100:.1f}%",
            'best': f"{best_acc*100:.1f}%"
        })

    # Print weight per epoch
    if leigh_row is not None:
        suffix = "  <- converged" if epoch == EPOCHS else ""
        for hid, (name, cnt, spec) in TRACK_SYMPTOMS.items():
            if hid in s_idx:
                w = model.coef_[leigh_row, s_idx[hid]]
                tqdm.write(f"  Epoch {epoch:>2}: W[ORPHA:506][{name.strip():<20}] = {w:+.5f}{suffix}")

print(f"\n  Final accuracy: {best_acc*100:.1f}%")

# STEP 6. Inspect learned weights
print("\n[STEP 5] Learned weight sample...")

coef_df = pd.DataFrame(model.coef_, index=le.classes_, columns=symptom_list)

for disease_code in le.classes_[:3]:
    row = coef_df.loc[disease_code].sort_values(ascending=False)
    print(f"\n  [{disease_code}] Top 5 symptoms for diagnosis")
    for sym_id, w in row.head(5).items():
        name = id_to_name.get(sym_id, sym_id)[:48]
        bar = chr(9608) * min(int(abs(w) * 5), 20)
        print(f"    {name:<48}  {w:+.3f}  {bar}")

# STEP 7. Save
print("\n[STEP 6] Saving model...")

with open(os.path.join(BASE_DIR, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)
with open(os.path.join(BASE_DIR, 'metadata.pkl'), 'wb') as f:
    pickle.dump({
        'symptom_list': symptom_list,
        'disease_list': disease_list,
        'id_to_name':   id_to_name,
        'name_to_id':   name_to_id,
        's_idx':        s_idx,
        'le':           le,
    }, f)

print("  Saving weight CSV...")
coef_named = coef_df.rename(columns=id_to_name)
coef_named.index.name = 'disease_id'
coef_named.reset_index().to_csv(
    os.path.join(BASE_DIR, 'symptom_weights.csv'),
    index=False, encoding='utf-8-sig'
)
print("  Done: model.pkl / metadata.pkl / symptom_weights.csv")

# STEP 8. Diagnosis function

def diagnose(input_symptoms: list, top_n: int = 10):
    """
    Input a list of symptom names -> returns disease probabilities
    e.g.) diagnose(['Seizure', 'Ataxia', 'Hearing loss'])
    """
    matched, unmatched = [], []
    for s in input_symptoms:
        candidates = sorted(
            [(name, hid) for name, hid in name_to_id.items()
             if s.lower() in name.lower() and hid in s_idx],
            key=lambda x: len(x[0])
        )
        if candidates:
            matched.append(candidates[0])
        else:
            unmatched.append(s)

    if unmatched:
        print(f"No match found: {unmatched}")
    if not matched:
        print("No symptoms matched")
        return None

    print("  ✓ Matched symptoms:")
    for name, hid in matched:
        print(f"     - {name}  ({hid})")

    vec = np.zeros((1, len(symptom_list)), dtype=np.float32)
    for _, hid in matched:
        vec[0, s_idx[hid]] = 1.0

    proba = model.predict_proba(vec)[0]
    top_idx = np.argsort(proba)[::-1][:top_n]
    return pd.DataFrame([
        {'disease_id': le.classes_[i], 'probability(%)': round(proba[i] * 100, 4)}
        for i in top_idx
    ])


# Diagnosis test 
print("\n" + "=" * 60)
print("  Diagnosis Test")
print("=" * 60)

for title, symptoms in [
    ("Seizure + Intellectual disability + Hypotonia",
     ['Seizure', 'Intellectual disability', 'Hypotonia']),
    ("Visual impairment + Nystagmus + Hypopigmentation",
     ['Visual impairment', 'Nystagmus', 'Hypopigmentation']),
    ("Congenital heart defect + Facial dysmorphism + Short stature",
     ['Congenital heart defect', 'Facial dysmorphism', 'Short stature']),
]:
    print(f"\n{'─'*60}\n  Symptoms: {title}\n{'─'*60}")
    result = diagnose(symptoms)
    if result is not None:
        print(f"\n  {'Disease ID':<22}  {'Prob%':>8}   Bar chart")
        print(f"  {'─'*52}")
        for _, row in result.iterrows():
            bar = chr(9608) * max(1, int(row['probability(%)'] * 2))
            print(f"  {row['disease_id']:<22}  {row['probability(%)']:>7.3f}%   {bar}")

print("\nDone! To run diagnosis manually:")
print("   result = diagnose(['Seizure', 'Ataxia', 'Hearing loss'])")
print("   print(result)")
