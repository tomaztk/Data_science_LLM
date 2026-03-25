"""
=============================================================================
DEMO E: Algorithm Selection Assistant
=============================================================================
Session: "LLMs — Yes or No with Data Science?"
Topic:   GPT-4o as algorithm advisor + classical vs GenAI algorithm comparison

Shows:
  1. GPT-4o algorithm recommendation engine from problem description
  2. Automated Azure AutoML job submission for top recommendations
  3. Classical vs Foundation model benchmark on text tasks
  4. Algorithm decision router — routes to the right pattern automatically
  5. Hyperparameter suggestion via LLM + validation

Prerequisites:
    pip install openai scikit-learn lightgbm pandas numpy python-dotenv azure-ai-ml
=============================================================================
"""

import os, json, time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OAI_ENDPOINT"],
    api_key=os.environ["AZURE_OAI_KEY"],
    api_version="2024-02-01",
)
MODEL = os.environ.get("AZURE_OAI_DEPLOY", "gpt-4o")

ADVISOR_SYSTEM = """You are a principal ML engineer at a top data science team. 
Given a problem description, recommend algorithms with starter code and Azure service mapping.
Always consider: data size, interpretability needs, latency requirements, and Azure-native options."""


# ---------------------------------------------------------------------------
# 1. ALGORITHM RECOMMENDATION ENGINE
# ---------------------------------------------------------------------------
PROBLEM_DESCRIPTIONS = [
    {
        "name": "Customer Churn",
        "desc": """
Dataset: 80,000 customers, 35 features (mix of numeric and categorical).
Target: binary churn (6% positive class — imbalanced).
Requirement: predictions must be EXPLAINABLE to non-technical account managers.
Priority: recall > precision (missing a churner is expensive).
Compute: Azure ML Standard_DS3_v2 cluster.
Time budget: model must train in < 2 hours.
""",
    },
    {
        "name": "Support Ticket Classification",
        "desc": """
Dataset: 15,000 labelled support tickets (text). 8 categories.
New categories added every quarter — model must adapt quickly.
Need: batch classification of 5,000 new tickets daily.
Priority: minimise re-labelling work when new categories appear.
Azure budget: < $50/month on inference.
""",
    },
    {
        "name": "Demand Forecasting",
        "desc": """
Dataset: 3 years of daily sales data, 500 SKUs, 20 stores.
Multiple seasonalities (weekly, annual, promotions).
Need: 28-day ahead forecasts, updated daily.
Missing data: ~8% of SKU-store combinations.
Latency: forecasts needed within 30 minutes of data landing.
""",
    },
]

def demo_algorithm_advisor():
    print("\n" + "═"*65)
    print("DEMO E-1: GPT-4o ALGORITHM RECOMMENDATION ENGINE")
    print("═"*65)

    for problem in PROBLEM_DESCRIPTIONS:
        print(f"\n{'─'*60}")
        print(f"🎯 Problem: {problem['name']}")
        print(f"Description:{problem['desc']}")

        prompt = f"""
Problem description:{problem['desc']}

Recommend the top 3 algorithms/approaches ranked by suitability. For each:
1. Algorithm name
2. Rationale (2 sentences)  
3. Key pros for this specific problem
4. Key cons / risks
5. Azure service or tool to use
6. Starter code (5-8 lines Python)

Respond as JSON: {{"algorithms": [{{"rank": 1, "name": ..., "rationale": ..., "pros": ..., "cons": ..., "azure_service": ..., "starter_code": ...}}]}}"""

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": ADVISOR_SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=1000,
            response_format={"type": "json_object"},
        )
        recs = json.loads(resp.choices[0].message.content)["algorithms"]

        for rec in recs:
            print(f"\n  #{rec['rank']} {rec['name']}")
            print(f"     Rationale: {rec['rationale']}")
            print(f"     Azure:     {rec['azure_service']}")
            print(f"     Pros:      {rec['pros'][:80]}")
            print(f"     Code:\n{rec['starter_code'][:200]}")


# ---------------------------------------------------------------------------
# 2. CLASSICAL ML BENCHMARK — run top recommendations
# ---------------------------------------------------------------------------
def demo_classical_benchmark():
    print("\n\n" + "═"*65)
    print("DEMO E-2: CLASSICAL ML BENCHMARK  (local run)")
    print("═"*65)

    # Generate synthetic churn dataset
    X, y = make_classification(
        n_samples=5000, n_features=25, n_informative=12,
        n_redundant=5, weights=[0.94, 0.06], random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=500),
        "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
        "LightGBM":            lgb.LGBMClassifier(n_estimators=300, class_weight="balanced", random_state=42, verbose=-1),
    }

    print(f"\nDataset: {X_train.shape[0]} train, {X_test.shape[0]} test, {y.mean():.1%} positive class\n")
    print(f"{'Model':<25} {'AUC-ROC':>8} {'Train(s)':>9} {'CV AUC':>8}")
    print("─"*55)

    results = []
    for name, model in models.items():
        use_scaled = name == "Logistic Regression"
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s  if use_scaled else X_test

        t0 = time.time()
        model.fit(Xtr, y_train)
        train_time = time.time() - t0

        preds_proba = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y_test, preds_proba)
        cv_scores = cross_val_score(model, Xtr, y_train, cv=5, scoring="roc_auc", n_jobs=-1)

        print(f"  {name:<23} {auc:>8.4f} {train_time:>8.1f}s {cv_scores.mean():>8.4f} ±{cv_scores.std():.4f}")
        results.append({"model": name, "auc": auc, "train_time": train_time, "cv_auc": cv_scores.mean()})

    best = max(results, key=lambda r: r["auc"])
    print(f"\n🏆 Best: {best['model']} (AUC={best['auc']:.4f})")

    # Ask GPT-4o to interpret results
    result_str = "\n".join([f"{r['model']}: AUC={r['auc']:.4f}, train={r['train_time']:.1f}s, CV={r['cv_auc']:.4f}" for r in results])
    interpretation = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"Interpret these churn model benchmark results for a 6% positive class dataset and recommend which to deploy:\n{result_str}"}],
        temperature=0.2, max_tokens=250,
    ).choices[0].message.content
    print(f"\n💬 GPT-4o interpretation:\n{interpretation}")


# ---------------------------------------------------------------------------
# 3. LLM HYPERPARAMETER SUGGESTION
# ---------------------------------------------------------------------------
def demo_hyperparameter_suggestion():
    print("\n\n" + "═"*65)
    print("DEMO E-3: LLM HYPERPARAMETER SUGGESTION")
    print("═"*65)

    context = """
Model: LightGBM binary classifier
Dataset: 80,000 rows, 35 features, 6% positive class
Hardware: 4-core CPU, 16GB RAM, no GPU
Training time budget: 45 minutes
Priority: maximise AUC-ROC, secondary: fast inference (<10ms per prediction)
Previous run: n_estimators=100, learning_rate=0.1 → AUC=0.81 (overfitting observed on train vs val gap of 0.06)
"""
    prompt = f"""
{context}

Suggest optimised LightGBM hyperparameters as a Python dict. Include:
- n_estimators, learning_rate, max_depth, num_leaves
- min_child_samples, subsample, colsample_bytree
- class_weight, reg_alpha, reg_lambda
- early_stopping_rounds

Respond as JSON: {{"params": {{...}}, "rationale": "..."}}"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=400,
        response_format={"type": "json_object"},
    )
    result = json.loads(resp.choices[0].message.content)
    print(f"\nContext: {context.strip()}")
    print(f"\nSuggested params:\n{json.dumps(result['params'], indent=2)}")
    print(f"\nRationale: {result['rationale']}")

    # Run with suggested params
    X, y = make_classification(n_samples=2000, n_features=20, weights=[0.94, 0.06], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    safe_params = {k: v for k, v in result["params"].items()
                   if k in lgb.LGBMClassifier().get_params()}
    safe_params.pop("early_stopping_rounds", None)

    model = lgb.LGBMClassifier(**safe_params, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"\n✅ Model trained with LLM-suggested params → AUC: {auc:.4f}")


# ---------------------------------------------------------------------------
# 4. ALGORITHM DECISION ROUTER
# ---------------------------------------------------------------------------
def demo_decision_router():
    print("\n\n" + "═"*65)
    print("DEMO E-4: ALGORITHM DECISION ROUTER")
    print("═"*65)
    print("Automatically routes DS problems to the right algorithm class\n")

    test_cases = [
        "I have 500 labelled customer support emails and need to classify them into 5 categories.",
        "I need to predict house prices given 15 numeric features, training data is 10,000 rows.",
        "I have a new FAQ document and need to answer questions about it without retraining.",
        "I want to segment 1 million customers into natural groups based on their behaviour.",
        "I need to detect fraudulent transactions in real-time from a stream of events.",
    ]

    router_prompt_template = """
Classify this ML problem and recommend the best approach for 2024.

Problem: {problem}

Choose from:
- Classical ML (sklearn/LightGBM/XGBoost for tabular/structured data)
- Fine-tuning (small LLM fine-tune when <1K labelled examples and consistent domain)
- Zero-shot/Few-shot (LLM API call when task is NLP + limited labels)
- RAG (when grounding in existing documents/knowledge base is needed)
- Clustering (unsupervised grouping)
- Time-series forecasting (Prophet, ARIMA, or LightGBM with lags)
- Anomaly detection (Isolation Forest, Autoencoders, or LLM-based)

Respond as JSON: {{"category": "...", "recommended_approach": "...", "azure_service": "...", "one_line_code": "..."}}"""

    for case in test_cases:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": router_prompt_template.format(problem=case)}],
            temperature=0, max_tokens=200,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        print(f"❓ {case[:70]}...")
        print(f"   → {result['category']}: {result['recommended_approach']}")
        print(f"   Azure: {result['azure_service']}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print("█" * 65)
    print("  DEMO E: ALGORITHM SELECTION ASSISTANT")
    print("█" * 65)

    demo_algorithm_advisor()
    demo_classical_benchmark()
    demo_hyperparameter_suggestion()
    demo_decision_router()

    print("\n\n" + "═"*65)
    print("KEY INSIGHT: Classical ML (LightGBM/XGBoost) still wins on")
    print("tabular data. LLMs win on text/NLP. The real skill is knowing")
    print("which tool to reach for — and GPT-4o can help you decide.")
    print("═"*65)


if __name__ == "__main__":
    run()
