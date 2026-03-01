"""
희귀질환 진단기 (정보 이득 기반 추가 증상 추천 포함)
=====================================================
[필요 파일] model.pkl, metadata.pkl, rare_diseases_genes.csv,
            phenotype_to_genes.txt
[실행 방법] python diagnose.py
"""

import pickle
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')  # 수치 경고 억제

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════
# 1. 모델 & 메타데이터 로드
# ══════════════════════════════════════════════════════
print("모델 로드 중...")
with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, 'metadata.pkl'), 'rb') as f:
    meta = pickle.load(f)

symptom_list = meta['symptom_list']
extra_cols   = meta.get('extra_cols', [])
id_to_name   = meta['id_to_name']
name_to_id   = meta['name_to_id']
s_idx        = meta['s_idx']
extra_idx    = meta.get('extra_idx', {})
le           = meta['le']
disease_list = list(le.classes_)
d_idx        = {d: i for i, d in enumerate(disease_list)}
n_features   = len(symptom_list) + len(extra_cols)

# ══════════════════════════════════════════════════════
# 2. 유전자 데이터 로드
# ══════════════════════════════════════════════════════
genes_df = pd.read_csv(os.path.join(BASE_DIR, 'rare_diseases_genes.csv'))
genes_df['disease_id'] = 'ORPHA:' + genes_df['OrphaCode'].astype(str)

def get_priority(t):
    if 'Disease-causing' in str(t): return 1
    if 'Major susceptibility' in str(t): return 2
    if 'Candidate' in str(t): return 3
    if 'Biomarker' in str(t): return 4
    return 5

def get_test_label(t):
    if 'loss of function'        in str(t): return '기능 소실 변이 검사'
    if 'gain of function'        in str(t): return '기능 획득 변이 검사'
    if 'somatic'                 in str(t): return '체세포 변이 검사'
    if 'Disease-causing germline' in str(t): return '생식세포 변이 검사'
    if 'Major susceptibility'    in str(t): return '감수성 유전자 검사'
    if 'Biomarker'               in str(t): return '바이오마커 검사'
    if 'Candidate'               in str(t): return '후보 유전자 검사 (참고용)'
    return '유전자 검사'

genes_df['priority']   = genes_df['AssociationType'].apply(get_priority)
genes_df['test_label'] = genes_df['AssociationType'].apply(get_test_label)

# ══════════════════════════════════════════════════════
# 3. 질환-증상 행렬 재구성 (정보 이득 계산용)
# ══════════════════════════════════════════════════════
print("질환-증상 행렬 재구성 중...")
X_matrix = np.zeros((len(disease_list), len(symptom_list)), dtype=np.float32)

hpo_path = os.path.join(BASE_DIR, 'phenotype_to_genes.txt')
if os.path.exists(hpo_path):
    hpo_raw = pd.read_csv(hpo_path, sep='\t')
    hpo = (hpo_raw[hpo_raw['disease_id'].str.startswith('ORPHA')]
           [['hpo_id', 'disease_id']].drop_duplicates())
    for row in hpo.itertuples(index=False):
        di = d_idx.get(row.disease_id)
        si = s_idx.get(row.hpo_id)
        if di is not None and si is not None:
            X_matrix[di, si] = 1.0
    print(f"행렬 재구성 완료\n")
else:
    print("⚠ phenotype_to_genes.txt 없음 — 추가 증상 추천 비활성화\n")

print(f"로드 완료 — 질환 {len(disease_list)}개 / 증상 {len(symptom_list)}개\n")

# ══════════════════════════════════════════════════════
# 선택지 정의
# ══════════════════════════════════════════════════════
AGE_OPTIONS = {
    '1': ('onset_Antenatal',  '태아기 (Antenatal)'),
    '2': ('onset_Neonatal',   '신생아기 (Neonatal, 0~1개월)'),
    '3': ('onset_Infancy',    '영아기 (Infancy, 1~12개월)'),
    '4': ('onset_Childhood',  '소아기 (Childhood, 1~10세)'),
    '5': ('onset_Adolescent', '청소년기 (Adolescent, 10~19세)'),
    '6': ('onset_Adult',      '성인기 (Adult, 19세 이상)'),
    '7': ('onset_Elderly',    '노년기 (Elderly, 60세 이상)'),
}
INH_OPTIONS = {
    '1': ('inh_AR',    '상염색체 열성 (Autosomal recessive)'),
    '2': ('inh_AD',    '상염색체 우성 (Autosomal dominant)'),
    '3': ('inh_XLR',   'X연관 열성 (X-linked recessive)'),
    '4': ('inh_XLD',   'X연관 우성 (X-linked dominant)'),
    '5': ('inh_Mito',  '미토콘드리아 유전 (Mitochondrial)'),
    '6': ('inh_Multi', '다인자 유전 (Multigenic)'),
}

# ══════════════════════════════════════════════════════
# 유틸 함수
# ══════════════════════════════════════════════════════
def search_symptom(keyword, max_results=10):
    results = sorted(
        [(name, hid) for name, hid in name_to_id.items()
         if keyword.lower() in name.lower() and hid in s_idx],
        key=lambda x: len(x[0])
    )
    print(f"\n  '{keyword}' 검색 결과 (상위 {max_results}개):")
    for name, hid in results[:max_results]:
        print(f"    [{hid}]  {name}")

def recommend_tests(disease_id):
    rows = genes_df[
        (genes_df['disease_id'] == disease_id) &
        (genes_df['AssociationStatus'] == 'Assessed')
    ].sort_values('priority')
    if rows.empty:
        print("    검사 정보 없음")
        return
    for _, row in rows.iterrows():
        mark = '🔴' if row['priority'] == 1 else ('🟡' if row['priority'] == 2 else '⚪')
        print(f"    {mark} {row['GeneSymbol']:<10}  {row['test_label']}")
        print(f"         유전자명: {row['GeneName']}")

def safe_predict_proba(vec):
    """수치 오버플로우 방지하여 확률 예측"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw = model.decision_function(vec)[0]
    # softmax로 직접 계산 (오버플로우 방지)
    raw = raw - np.max(raw)
    exp = np.exp(np.clip(raw, -500, 500))
    return exp / exp.sum()

# ══════════════════════════════════════════════════════
# 정보 이득 계산 및 추가 증상 추천
# ══════════════════════════════════════════════════════
def entropy(p):
    p = p[p > 1e-10]
    return float(-np.sum(p * np.log2(p)))

def recommend_next_symptoms(current_vec, current_proba, already_asked_ids, top_k=5, candidate_n=60):
    """
    정보 이득이 가장 높은 추가 증상 top_k개 반환.

    정보 이득 = 현재 엔트로피 - (p_yes * H_yes + p_no * H_no)
      H_yes : 해당 증상이 있다고 했을 때의 엔트로피
      H_no  : 해당 증상이 없다고 했을 때의 엔트로피
      p_yes : 현재 상위 질환들에서 이 증상이 나타나는 비율 (사전확률)
    """
    current_h = entropy(current_proba)

    # 상위 10개 질환에서 자주 나타나는 증상을 후보로
    top10_idx = np.argsort(current_proba)[::-1][:10]
    sym_freq  = X_matrix[top10_idx].sum(axis=0)  # 각 증상이 상위 질환 중 몇 개에 있나

    # 이미 입력된 증상 제외, 빈도 높은 순으로 candidate_n개
    excluded = set(already_asked_ids)
    candidates = [
        symptom_list[i] for i in np.argsort(sym_freq)[::-1]
        if symptom_list[i] not in excluded and sym_freq[i] > 0
    ][:candidate_n]

    results = []
    for hid in candidates:
        si = s_idx[hid]

        vec_yes = current_vec.copy()
        vec_yes[0, si] = 1.0
        proba_yes = safe_predict_proba(vec_yes)

        vec_no = current_vec.copy()
        vec_no[0, si] = 0.0
        proba_no = safe_predict_proba(vec_no)

        # 사전확률: 상위 10개 질환 중 이 증상이 있는 비율
        p_yes = float(sym_freq[si]) / len(top10_idx)
        p_no  = 1.0 - p_yes

        ig = current_h - (p_yes * entropy(proba_yes) + p_no * entropy(proba_no))
        if ig <= 0:
            continue

        top1        = int(np.argmax(current_proba))
        prob_now    = float(current_proba[top1]) * 100
        prob_yes_t1 = float(proba_yes[top1]) * 100
        prob_no_t1  = float(proba_no[top1])  * 100

        results.append((hid, id_to_name.get(hid, hid), ig,
                        prob_yes_t1, prob_no_t1, prob_now))

    results.sort(key=lambda x: -x[2])
    return results[:top_k]

def print_next_symptom_recommendations(current_vec, current_proba, already_asked_ids):
    current_h = entropy(current_proba)
    top1      = int(np.argmax(current_proba))
    top1_prob = float(current_proba[top1]) * 100

    print(f"\n  {'─'*58}")
    print(f"  📋 추가로 확인하면 좋을 증상 (정보 이득 순)")
    print(f"  현재 불확실성: {current_h:.2f}  |  1위 질환 확률: {top1_prob:.3f}%")
    print(f"  {'─'*58}")

    recs = recommend_next_symptoms(current_vec, current_proba, already_asked_ids)

    if not recs:
        print("  (추천할 증상 없음)")
        return

    for rank, (hid, name, ig, prob_yes, prob_no, prob_now) in enumerate(recs, 1):
        stars    = min(5, max(1, int(ig * 10)))
        star_str = '★' * stars + '☆' * (5 - stars)
        d_yes    = prob_yes - prob_now
        d_no     = prob_no  - prob_now

        print(f"\n  [{rank}위] {name}")
        print(f"         HPO ID     : {hid}")
        print(f"         정보 이득  : {star_str}  ({ig:.4f})")
        print(f"         있다고 하면: {prob_now:.3f}% → {prob_yes:.3f}%  "
              f"({'%+.3f' % d_yes}%p)")
        print(f"         없다고 하면: {prob_now:.3f}% → {prob_no:.3f}%  "
              f"({'%+.3f' % d_no}%p)")

    print(f"\n  {'─'*58}")
    print(f"  ※ ★ 이 많을수록 이 증상을 확인했을 때 진단이 크게 좁혀집니다.")

# ══════════════════════════════════════════════════════
# 진단 메인 함수
# ══════════════════════════════════════════════════════
def diagnose(input_symptoms, onset_keys=None, inh_keys=None, top_n=5):
    matched, unmatched = [], []
    for s in input_symptoms:
        candidates = sorted(
            [(name, hid) for name, hid in name_to_id.items()
             if s.lower() in name.lower() and hid in s_idx],
            key=lambda x: len(x[0])
        )
        if candidates: matched.append(candidates[0])
        else:          unmatched.append(s)

    if unmatched:
        print(f"\n  ⚠  찾지 못한 증상: {unmatched}")
    if not matched:
        print("  ✗  매칭된 증상이 없습니다.")
        return

    print("\n  입력된 증상:")
    for name, hid in matched:
        print(f"    - {name}  ({hid})")
    if onset_keys: print("  발병 나이:", ', '.join(onset_keys))
    if inh_keys:   print("  유전 방식:", ', '.join(inh_keys))

    # 입력 벡터
    vec = np.zeros((1, n_features), dtype=np.float64)  # float64로 오버플로우 방지
    for _, hid in matched:
        vec[0, s_idx[hid]] = 1.0
    offset = len(symptom_list)
    for key in (onset_keys or []):
        if key in extra_idx: vec[0, offset + extra_idx[key]] = 1.0
    for key in (inh_keys or []):
        if key in extra_idx: vec[0, offset + extra_idx[key]] = 1.0

    proba   = safe_predict_proba(vec)
    top_idx = np.argsort(proba)[::-1][:top_n]

    # ── 진단 결과 ─────────────────────────────────────
    for rank, idx in enumerate(top_idx, 1):
        disease = le.classes_[idx]
        prob    = proba[idx] * 100
        code    = disease.replace("ORPHA:", "")
        bar     = chr(9608) * max(1, int(prob * 2))

        print(f"\n  {'='*58}")
        print(f"  [{rank}위]  {disease}   {prob:.3f}%  {bar}")
        print(f"  {'─'*58}")
        name_row = genes_df[genes_df['disease_id'] == disease]['DiseaseName']
        if not name_row.empty:
            print(f"  질환명: {name_row.values[0]}")
        print(f"  정보  : https://www.orpha.net/en/disease/detail/{code}")
        print(f"  권장 유전자 검사:")
        recommend_tests(disease)

    print(f"\n  {'='*58}")
    print(f"  ※ 🔴 확진 검사  🟡 위험도 검사  ⚪ 참고용 검사")
    print(f"  ※ 이 결과는 참고용이며 반드시 전문의와 상담하세요.")

    # ── 추가 증상 추천 ────────────────────────────────
    already = {hid for _, hid in matched}
    print_next_symptom_recommendations(vec, proba, already)

# ══════════════════════════════════════════════════════
# 대화형 루프
# ══════════════════════════════════════════════════════
def run():
    print("=" * 60)
    print("  희귀질환 진단기  (증상 + 임상정보 + 추가 증상 추천)")
    print("=" * 60)
    print("""
  명령어:
    증상 입력  →  Seizure, Ataxia, Hearing loss
    증상 검색  →  search:heart
    종료       →  q
    """)

    while True:
        user_input = input("증상 입력 > ").strip()
        if not user_input: continue
        if user_input.lower() == 'q':
            print("종료합니다."); break
        if user_input.lower().startswith('search:'):
            search_symptom(user_input.split(':', 1)[1].strip())
            print(); continue

        symptoms = [s.strip() for s in user_input.split(',')]

        print("\n  발병 나이를 선택하세요 (모르면 Enter):")
        for k, (_, label) in AGE_OPTIONS.items():
            print(f"    {k}) {label}")
        age_input = input("  번호 > ").strip()
        onset_keys = [AGE_OPTIONS[n][0] for n in age_input.replace(',', ' ').split()
                      if n in AGE_OPTIONS]

        print("\n  유전 방식을 선택하세요 (모르면 Enter):")
        for k, (_, label) in INH_OPTIONS.items():
            print(f"    {k}) {label}")
        inh_input = input("  번호 > ").strip()
        inh_keys = [INH_OPTIONS[n][0] for n in inh_input.replace(',', ' ').split()
                    if n in INH_OPTIONS]

        diagnose(symptoms, onset_keys or None, inh_keys or None)
        print()

if __name__ == '__main__':
    run()
