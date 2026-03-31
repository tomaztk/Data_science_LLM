"""
=============================================================================
DEMO F: SHAP + LLM Explainability Pipeline
=============================================================================
Session: "LLMs — Yes or No with Data Science?"
Topic:   Classical XAI (SHAP/LIME) enhanced by LLM narrative generation

Shows:
  1. SHAP global and local explanations (classical XAI)
  2. LLM narration of SHAP for stakeholders
  3. LLM-as-judge: evaluate narrative quality at scale
  4. Azure Responsible AI Dashboard metrics (simulated)
  5. LLM groundedness evaluation for RAG (GenAI-specific XAI)
  6. Counterfactual explanation generation via LLM

Prerequisites:
    pip install shap lightgbm scikit-learn openai pandas numpy matplotlib python-dotenv
=============================================================================
"""

import os
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

load_dotenv()

# Use Azure AD token instead of API key
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OAI_ENDPOINT"],
    azure_ad_token_provider=token_provider,
    api_version="2024-02-01",
)

MODEL = os.environ.get("AZURE_OAI_DEPLOY", "gpt-4o-mini")

FEATURE_NAMES = [
    "tenure_months", "monthly_charges", "support_tickets_90d", "feature_usage_score",
    "logins_last_30d", "nps_score", "avg_order_value", "days_since_last_login",
    "plan_enterprise", "payment_failures_6m", "email_open_rate", "contract_remaining_days",
]


# ---------------------------------------------------------------------------
# 0. Train model and compute SHAP
# ---------------------------------------------------------------------------
def setup_model():
    np.random.seed(42)
    X, y = make_classification(
        n_samples=3000, n_features=len(FEATURE_NAMES),
        n_informative=8, n_redundant=2, weights=[0.93, 0.07], random_state=42
    )
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["churned"] = y

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURE_NAMES], df["churned"], test_size=0.2, stratify=df["churned"], random_state=42
    )

    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                class_weight="balanced", random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"\n Model trained — AUC-ROC: {auc:.4f} | Positive rate: {y_test.mean():.1%}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    preds_proba = model.predict_proba(X_test)[:, 1]

    return model, X_test, y_test, shap_values, preds_proba


# ---------------------------------------------------------------------------
# 1. CLASSICAL SHAP — global & local
# ---------------------------------------------------------------------------
def demo_classical_shap(shap_values, X_test, preds_proba):
    print("\n" + "═"*65)
    print("DEMO F-1: CLASSICAL SHAP — Global & Local Explanations")
    print("═"*65)

    # Global importance
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "mean_abs_shap": mean_shap
    }).sort_values("mean_abs_shap", ascending=False)

    print("\nGlobal Feature Importance (mean |SHAP|):")
    print(importance_df.to_string(index=False))

    # Save SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False, max_display=12)
    plt.title("SHAP Summary — Churn Prediction Model", fontsize=13)
    plt.tight_layout()
    plt.savefig("shap_summary_f.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("\n Saved: shap_summary_f.png")

    # Local explanation for a high-risk customer
    high_risk_idx = np.argmax(preds_proba)
    print(f"\nHigh-risk customer (idx={high_risk_idx}, p={preds_proba[high_risk_idx]:.1%}):")
    local_shap = dict(zip(FEATURE_NAMES, shap_values.values[high_risk_idx]))
    sorted_local = sorted(local_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
    for feat, val in sorted_local:
        direction = "↑ risk" if val > 0 else "↓ risk"
        print(f"  {feat:<30}: SHAP={val:+.4f}  ({direction})")


# ---------------------------------------------------------------------------
# 2. LLM NARRATION OF SHAP
# ---------------------------------------------------------------------------
def demo_llm_shap_narration(shap_values, X_test, preds_proba):
    print("\n\n" + "═"*65)
    print("DEMO F-2: LLM NARRATION OF SHAP VALUES")
    print("═"*65)
    print("Translating model explanations into business language...\n")

    # Select top 5 at-risk customers
    top_risk_idx = np.argsort(preds_proba)[-5:]
    narratives = []

    for i, idx in enumerate(reversed(top_risk_idx), 1):
        risk = preds_proba[idx]
        shap_row = dict(zip(FEATURE_NAMES, shap_values.values[idx]))
        feature_vals = dict(zip(FEATURE_NAMES, X_test.iloc[idx].values))

        # Build context for LLM
        top_factors = sorted(shap_row.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        factor_text = "\n".join([
            f"  {feat} = {feature_vals[feat]:.1f}  →  SHAP={val:+.3f} ({'risk driver' if val > 0 else 'protective factor'})"
            for feat, val in top_factors
        ])

        prompt = f"""Write a 2-sentence churn risk explanation for a customer account manager.

Customer churn probability: {risk:.0%}
Top model factors (positive SHAP = increases churn risk):
{factor_text}

Rules:
- NO ML jargon (no "SHAP", "model", "feature", "coefficient")
- Be specific about what the customer is doing (or not doing)
- End with ONE concrete, actionable recommendation for the account manager
- Tone: professional, empathetic, data-backed"""

        narrative = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=150,
        ).choices[0].message.content

        narratives.append({"idx": idx, "risk": risk, "narrative": narrative})
        print(f"Customer #{i}  |  Risk: {risk:.0%}")
        print(textwrap.fill(narrative, 70, initial_indent="  ", subsequent_indent="  "))
        print()

    return narratives


# ---------------------------------------------------------------------------
# 3. LLM-AS-JUDGE: Evaluate Narrative Quality at Scale
# ---------------------------------------------------------------------------
def demo_llm_as_judge(narratives: list[dict]):
    print("\n\n" + "═"*65)
    print("DEMO F-3: LLM-AS-JUDGE — Evaluate Narrative Quality")
    print("═"*65)
    print("Using GPT-4o-mini to score GPT-4o-generated narratives...\n")

    judge_system = (
        "You are a quality evaluator for AI-generated churn risk explanations. "
        "Score each explanation on: (1) Clarity for non-technical reader (2) Actionability (3) Accuracy/specificity. "
        "Return JSON: {\"clarity\": 0-10, \"actionability\": 0-10, \"specificity\": 0-10, \"overall\": 0-10, \"feedback\": \"one sentence\"}"
    )

    scores = []
    for n in narratives:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": judge_system},
                {"role": "user", "content": f"Rate this churn explanation (risk={n['risk']:.0%}):\n{n['narrative']}"},
            ],
            temperature=0, max_tokens=150,
            response_format={"type": "json_object"},
        )
        score = json.loads(resp.choices[0].message.content)
        scores.append(score)
        print(f"Risk {n['risk']:.0%}  |  Overall: {score['overall']}/10  |  Feedback: {score['feedback']}")

    avg_overall = sum(s["overall"] for s in scores) / len(scores)
    avg_clarity = sum(s["clarity"] for s in scores) / len(scores)
    avg_action  = sum(s["actionability"] for s in scores) / len(scores)
    print(f"\n Mean scores — Overall: {avg_overall:.1f}  Clarity: {avg_clarity:.1f}  Actionability: {avg_action:.1f}")
    print(" Use this as a prompt quality gate before sending narratives to account managers.")


# ---------------------------------------------------------------------------
# 4. COUNTERFACTUAL EXPLANATION
# ---------------------------------------------------------------------------
def demo_counterfactuals(shap_values, X_test, preds_proba, model):
    print("\n\n" + "═"*65)
    print("DEMO F-4: LLM-GENERATED COUNTERFACTUAL EXPLANATIONS")
    print("═"*65)
    print("'What would need to change for this customer NOT to churn?'\n")

    high_risk_idx = np.argmax(preds_proba)
    customer_data = dict(zip(FEATURE_NAMES, X_test.iloc[high_risk_idx].values))
    risk = preds_proba[high_risk_idx]

    shap_row = dict(zip(FEATURE_NAMES, shap_values.values[high_risk_idx]))
    top_drivers = sorted(
        [(f, v) for f, v in shap_row.items() if v > 0],
        key=lambda x: x[1], reverse=True
    )[:4]

    prompt = f"""
A customer has a {risk:.0%} churn probability. Their key risk drivers are:
{chr(10).join(f"  - {f}: current value = {customer_data[f]:.1f}, risk contribution = {v:+.3f}" for f,v in top_drivers)}

Generate 3 concrete counterfactual interventions:
For each, specify:
1. What needs to change (in plain language, no ML terms)
2. How much it would need to change (approximate)
3. Estimated churn risk reduction (High/Medium/Low)
4. Who owns this action (account manager / product team / pricing team)

Format as a numbered list."""

    counterfactuals = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=400,
    ).choices[0].message.content

    print(f"Customer risk: {risk:.0%}")
    print(f"Top risk drivers: {', '.join(f for f,_ in top_drivers)}\n")
    print("Counterfactual interventions:")
    print(counterfactuals)


# ---------------------------------------------------------------------------
# 5. GROUNDEDNESS EVALUATION (GenAI-specific XAI)
# ---------------------------------------------------------------------------
def demo_groundedness_eval():
    print("\n\n" + "═"*65)
    print("DEMO F-5: GROUNDEDNESS EVALUATION FOR RAG OUTPUTS")
    print("═"*65)
    print("Evaluating if LLM answers are actually grounded in retrieved context...\n")

    test_pairs = [
        {
            "context": "Azure AI Foundry supports over 1,800 models from Microsoft, OpenAI, Meta, Mistral, and others. Models can be deployed to managed endpoints or consumed via serverless APIs.",
            "question": "How many models does Azure AI Foundry support?",
            "answer_good": "Azure AI Foundry supports over 1,800 models from various providers including Microsoft, OpenAI, Meta, and Mistral.",
            "answer_hallucinated": "Azure AI Foundry supports over 5,000 models and also includes pre-built industry solutions for healthcare and finance.",
        },
        {
            "context": "RAG (Retrieval-Augmented Generation) combines a retrieval system with a generative model. Azure AI Search supports both keyword (BM25) and vector (HNSW) retrieval.",
            "question": "What retrieval methods does Azure AI Search support?",
            "answer_good": "Azure AI Search supports BM25 keyword retrieval and HNSW vector retrieval, enabling hybrid search for RAG systems.",
            "answer_hallucinated": "Azure AI Search uses GPT-4 to intelligently rank results using reinforcement learning from human feedback.",
        },
    ]

    eval_prompt_template = """
Rate the groundedness of this answer based ONLY on the provided context.
Groundedness (0-10): Is every claim in the answer supported by the context?
Flag any claims that are NOT in the context as hallucinations.

Context: {context}
Question: {question}
Answer: {answer}

Respond as JSON: {{"groundedness": 0-10, "hallucinations": ["list of unsupported claims"], "verdict": "Grounded | Partially Grounded | Hallucinated"}}"""

    for pair in test_pairs:
        print(f"\n Context: {pair['context'][:80]}...")
        print(f"  Question: {pair['question']}\n")

        for label, answer in [(" Good answer", pair["answer_good"]), (" Hallucinated", pair["answer_hallucinated"])]:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": eval_prompt_template.format(
                    context=pair["context"], question=pair["question"], answer=answer
                )}],
                temperature=0, max_tokens=200,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            print(f"  {label}: Groundedness={result['groundedness']}/10  |  {result['verdict']}")
            if result["hallucinations"]:
                print(f"  Hallucinations detected: {result['hallucinations']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print("█" * 65)
    print("  DEMO F: SHAP + LLM EXPLAINABILITY PIPELINE")
    print("█" * 65)

    print("\nSetting up model and SHAP explainer...")
    model, X_test, y_test, shap_values, preds_proba = setup_model()

    demo_classical_shap(shap_values, X_test, preds_proba)
    narratives = demo_llm_shap_narration(shap_values, X_test, preds_proba)
    demo_llm_as_judge(narratives)
    demo_counterfactuals(shap_values, X_test, preds_proba, model)
    demo_groundedness_eval()

    print("\n\n" + "═"*65)
    print("KEY INSIGHT: SHAP answers 'what drove this prediction?'")
    print("LLMs answer 'what does that mean for a human?'")
    print("Combined = full explainability stack for modern DS teams.")
    print("Groundedness eval = the new 'model evaluation' for GenAI.")
    print("═"*65)


if __name__ == "__main__":
    run()
