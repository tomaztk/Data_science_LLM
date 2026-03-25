"""
=============================================================================
DEMO C: Multi-Language LLM Integration
=============================================================================
Session: "LLMs — Yes or No with Data Science?"
Topic:   Python, R-equivalent, SQL — same DS tasks in multiple languages via LLMs

Shows:
  1. Python: classification, extraction, structured output
  2. R-equivalent prompting patterns (translated to Python for runability)
  3. Text-to-SQL: natural language → SQL query via Azure OpenAI
  4. Prompt Engineering patterns (zero-shot / few-shot / CoT / structured)
  5. Language trend analysis — GPT-4o analyses SO survey data

Prerequisites:
    pip install openai pandas numpy python-dotenv
=============================================================================
"""

import os, json, textwrap
import pandas as pd
import numpy as np
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OAI_ENDPOINT"],
    api_key=os.environ["AZURE_OAI_KEY"],
    api_version="2024-02-01",
)
MODEL = os.environ.get("AZURE_OAI_DEPLOY", "gpt-4o")

SYSTEM_DS = "You are a senior data scientist. Be concise and precise."


def ask(prompt: str, system: str = SYSTEM_DS, json_mode: bool = False, max_tokens: int = 600) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=max_tokens, **kwargs,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# 1. PYTHON PATTERNS — Zero-shot, Few-shot, CoT, Structured Output
# ---------------------------------------------------------------------------
SAMPLE_REVIEWS = [
    "The Azure ML SDK v2 is a massive improvement but the docs are still catching up.",
    "GitHub Copilot changed my workflow completely — I write comments, not code.",
    "Prompt Flow is confusing at first but incredibly powerful once you get it.",
    "Fine-tuning GPT-4 is way too expensive for most use cases. RAG is the answer.",
    "R support in Azure ML is fine but clearly Python is the first-class citizen.",
]

def demo_python_patterns():
    print("\n" + "═"*65)
    print("DEMO C-1: PYTHON PROMPTING PATTERNS")
    print("═"*65)

    # Zero-shot classification
    print("\n[Pattern 1] ZERO-SHOT — Sentiment + topic extraction")
    results = []
    for review in SAMPLE_REVIEWS:
        r = ask(f"Classify this DS practitioner review.\nSentiment: Positive/Neutral/Negative\nTopic: Azure-ML/Copilot/Prompting/Cost/Other\nReview: {review}\n\nRespond as JSON: {{\"sentiment\": ..., \"topic\": ...}}", json_mode=True)
        parsed = json.loads(r)
        results.append({"review": review[:55]+"...", **parsed})
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Few-shot classification
    print("\n[Pattern 2] FEW-SHOT — Intent classification with examples")
    examples = """
Text: "How do I reduce my Azure OpenAI token costs?" → billing
Text: "My embedding model returns null vectors" → technical
Text: "I want to cancel my subscription" → cancellation
Text: "Which model should I use for summarisation?" → advice
"""
    test_msgs = [
        "My Prompt Flow pipeline keeps timing out",
        "I need a refund for this month",
        "What's the difference between GPT-4o and GPT-4 Turbo?",
    ]
    print(f"\nFew-shot examples:\n{examples}")
    for msg in test_msgs:
        r = ask(f"{examples}\nText: \"{msg}\" →", max_tokens=20)
        print(f"  '{msg}' → {r.strip()}")

    # Chain of Thought
    print("\n[Pattern 3] CHAIN-OF-THOUGHT — Statistical analysis")
    problem = """
A data scientist runs A/B test: 
Group A (n=1200): 8.3% conversion.
Group B (n=1150): 9.1% conversion.
p-value = 0.043, alpha = 0.05.
Budget allows deploying only one version for 90 days.
"""
    r = ask(f"Solve step by step, reason through it, then give a clear recommendation:\n{problem}", max_tokens=400)
    print(f"\nProblem: {problem.strip()}")
    print(f"\nGPT-4o CoT reasoning:\n{textwrap.fill(r, 70)}")

    # Structured Output — JSON mode
    print("\n[Pattern 4] STRUCTURED OUTPUT — JSON mode for pipeline integration")
    text = "Customer John Smith, 45 years old, enterprise plan, high churn risk score 0.87, last login 32 days ago."
    r = ask(f"Extract structured data from: '{text}'\nReturn JSON with: name, age, plan, churn_risk_score, days_since_login", json_mode=True)
    data = json.loads(r)
    print(f"\nRaw text: {text}")
    print(f"Extracted: {json.dumps(data, indent=2)}")
    df_extracted = pd.DataFrame([data])
    print(f"\nDataFrame:\n{df_extracted.to_string(index=False)}")


# ---------------------------------------------------------------------------
# 2. TEXT-TO-SQL  (the SQL language renaissance)
# ---------------------------------------------------------------------------
DB_SCHEMA = """
Database: sales_dwh

Tables:
  customers(customer_id INT PK, name VARCHAR, plan VARCHAR, region VARCHAR, signup_date DATE, churned BOOL)
  orders(order_id INT PK, customer_id INT FK, amount DECIMAL, order_date DATE, product_category VARCHAR)
  support_tickets(ticket_id INT PK, customer_id INT FK, created_date DATE, resolved_date DATE, category VARCHAR, satisfaction_score INT)
  events(event_id INT PK, customer_id INT FK, event_type VARCHAR, event_date DATE, metadata JSONB)
"""

NL_QUERIES = [
    "Top 10 customers by total revenue in the last 90 days, showing their plan type and churn status",
    "Monthly churn rate by plan type for the past year, as a percentage",
    "Average days to resolve support tickets by category, only for enterprise customers",
    "Customers who placed an order within 30 days of opening a support ticket — potential churn recovery",
    "Weekly active users trend for the last 12 weeks, defined as customers with at least one event",
]

def demo_text_to_sql():
    print("\n\n" + "═"*65)
    print("DEMO C-2: TEXT-TO-SQL  (SQL Language Renaissance via LLMs)")
    print("═"*65)
    print(f"\nSchema:\n{DB_SCHEMA}")

    sql_system = (
        "You are a senior SQL analyst. Given a schema and a natural language question, "
        "write clean, optimised SQL (PostgreSQL syntax). "
        "Add a one-line comment explaining what the query does. Return only SQL."
    )

    for i, query in enumerate(NL_QUERIES, 1):
        print(f"\n{'─'*60}")
        print(f"[{i}] Natural language: {query}")
        sql = ask(query + f"\n\nSchema:\n{DB_SCHEMA}", system=sql_system, max_tokens=300)
        print(f"\nGenerated SQL:\n{sql}")


# ---------------------------------------------------------------------------
# 3. R-EQUIVALENT PATTERNS (translated to Python demonstrating the concepts)
# ---------------------------------------------------------------------------
def demo_r_equivalent():
    print("\n\n" + "═"*65)
    print("DEMO C-3: R-EQUIVALENT PATTERNS via LLM")
    print("═"*65)

    # Translate Python code to R
    python_code = '''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df = pd.read_csv("churn.csv")
X = df.drop("churned", axis=1).select_dtypes(include="number")
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
model = LogisticRegression(class_weight="balanced")
model.fit(X_train_s, y_train)
print(f"AUC: {roc_auc_score(y_test, model.predict_proba(X_test_s)[:,1]):.4f}")
'''
    r_code = ask(
        f"Translate this Python ML code to R using tidyverse, tidymodels, and recipes:\n```python\n{python_code}\n```\n"
        "Use recipes for preprocessing, parsnip for models, yardstick for metrics. Return only R code.",
        system="You are an expert R programmer who loves tidymodels.", max_tokens=500
    )
    print("\nOriginal Python → Translated to R (tidymodels):")
    print("\n--- Python ---")
    print(python_code.strip()[:400])
    print("\n--- R (generated by GPT-4o) ---")
    print(r_code)


# ---------------------------------------------------------------------------
# 4. Language trend analysis — GPT-4o on Stack Overflow data
# ---------------------------------------------------------------------------
SO_DATA = {
    "year": [2019, 2020, 2021, 2022, 2023, 2024],
    "python_ds_pct": [66, 70, 72, 75, 77, 80],
    "r_ds_pct":      [46, 44, 41, 38, 36, 33],
    "sql_pct":       [55, 57, 57, 60, 62, 65],
    "scala_pct":     [16, 15, 13, 11,  9,  7],
    "julia_pct":     [ 3,  4,  5,  6,  6,  7],
}

def demo_language_trends():
    print("\n\n" + "═"*65)
    print("DEMO C-4: LANGUAGE TREND ANALYSIS (GPT-4o as Analyst)")
    print("═"*65)

    df = pd.DataFrame(SO_DATA)
    print(f"\nStack Overflow DS Language Usage (%):\n{df.to_string(index=False)}")

    analysis = ask(f"""
Analyse this Stack Overflow developer survey data on programming language usage in data science.

Data:
{df.to_string(index=False)}

Provide:
1. The most significant trend and why it happened
2. Which language's trajectory most surprised you and why
3. What does the SQL trend tell us about LLMs and data access?
4. Prediction for 2026 — which language grows most and why?

Be analytical and specific.""", max_tokens=500)
    print(f"\nGPT-4o Analysis:\n{analysis}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print("█" * 65)
    print("  DEMO C: MULTI-LANGUAGE LLM INTEGRATION")
    print("█" * 65)

    demo_python_patterns()
    demo_text_to_sql()
    demo_r_equivalent()
    demo_language_trends()

    print("\n\n" + "═"*65)
    print("KEY INSIGHT: LLMs don't eliminate languages — they reduce")
    print("switching cost between them. Python stays #1 but SQL is")
    print("resurging via NL-to-SQL, and 'prompting' is a new language.")
    print("═"*65)


if __name__ == "__main__":
    run()
