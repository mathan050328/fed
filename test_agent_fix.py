import sys
sys.path.insert(0, '.')
from src.evaluation.xai_agent import ExplainableAIAgent

agent = ExplainableAIAgent(disease='heart')
test_input = {
    'age': 56, 'sex': 1, 'cp': 2, 'trestbps': 135, 'chol': 250,
    'fbs': 0, 'restecg': 1, 'thalach': 142, 'exang': 1,
    'oldpeak': 1.4, 'slope': 1, 'ca': 1, 'thal': 3
}

queries = [
    ('why is the risk score high?', 'Diagnosis'),
    ('how to reduce risk?', 'Plan'),
    ('what lab tests?', 'Labs'),
    ('is this federated learning private?', 'Privacy'),
    ('general question', 'Default'),
]

all_ok = True
for q, label in queries:
    try:
        result = agent.ask_agent(test_input, q)
        ok = isinstance(result, dict) and all(k in result for k in ['title', 'badge', 'content'])
        badge = result.get('badge', 'MISSING') if isinstance(result, dict) else type(result).__name__
        print(f"[{label}]: OK={ok}, badge={badge}")
        if not ok:
            all_ok = False
    except Exception as e:
        print(f"[{label}]: ERROR - {e}")
        all_ok = False

print()
print("ALL QUERIES OK:", all_ok)

# Also test diabetes and CKD
print("\n--- Testing diabetes ---")
agent2 = ExplainableAIAgent(disease='diabetes')
test_diab = {'pregnancies': 3, 'glucose': 148, 'blood_pressure': 82, 'skin_thickness': 30,
             'insulin': 130, 'bmi': 33.6, 'diabetes_pedigree': 0.62, 'age': 48}
r = agent2.ask_agent(test_diab, 'is this federated learning private?')
print("Diabetes Privacy query:", type(r).__name__, "- badge:", r.get('badge', 'MISSING'))

print("\n--- Testing CKD ---")
agent3 = ExplainableAIAgent(disease='ckd')
test_ckd = {'age': 58, 'bp': 85, 'sg': 1.015, 'al': 2, 'su': 1, 'bgr': 150, 'bu': 58,
            'sc': 2.4, 'sod': 134, 'pot': 4.8, 'hemo': 10.5, 'pcv': 33, 'wbcc': 8900,
            'rbcc': 3.8, 'htn': 1, 'dm': 1, 'cad': 0, 'appet': 0, 'pe': 1, 'ane': 0}
r = agent3.ask_agent(test_ckd, 'is this federated learning private?')
print("CKD Privacy query:", type(r).__name__, "- badge:", r.get('badge', 'MISSING'))
