"""
=============================================================================
DEMO B: IDE Copilot Productivity Benchmark
=============================================================================
Session: "LLMs — Yes or No with Data Science?"
Topic:   Simulating GitHub Copilot-style intent-to-code generation via API

Shows:
  1. Intent comment → full DS function (simulating Copilot)
  2. Existing code → /explain (simulating VS Code Copilot chat)
  3. Existing code → /tests (automated test generation)
  4. Existing code → /fix (bug detection and correction)
  5. Productivity timing comparison: manual vs AI-assisted

Prerequisites:
    pip install openai pandas python-dotenv

Base:
    base(3.12.7) - Conda env
=============================================================================
"""

import os, json, time, textwrap
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OAI_ENDPOINT"],
    api_key=os.environ["AZURE_OAI_KEY"],
    api_version="2024-02-01",
)
MODEL = os.environ.get("AZURE_OAI_DEPLOY", "gpt-4o")


def ask(system: str, user: str, max_tokens: int = 600) -> tuple[str, float]:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content, round(time.time() - t0, 2)


# ---------------------------------------------------------------------------
# 1. INTENT → CODE  (simulates GitHub Copilot inline completion)
# ---------------------------------------------------------------------------
INTENTS = [
    "# Load a CSV from 'data/sales.csv', drop rows where 'revenue' is null,\n# encode 'region' as dummy variables, and split into X_train, X_test (80/20 stratified by 'churned')",
    "# Train a LightGBM classifier on X_train, y_train with early stopping.\n# Return the model and print AUC-ROC on X_test.",
    "# Compute SHAP values for a fitted LightGBM model and plot the top 10\n# features as a horizontal bar chart. Save to 'shap_importance.png'.",
    "# Query Azure ML workspace to list all experiments, filter the last 7 days,\n# and return a DataFrame with columns: run_id, experiment, AUC, status.",
    "# Call Azure OpenAI to classify a list of customer support messages\n# as 'billing', 'technical', 'cancellation', or 'other'. Return a list of labels.",
]

COPILOT_SYSTEM = (
    "You are GitHub Copilot, an expert Python programmer. "
    "Given a comment describing what code should do, write clean, production-quality Python. "
    "Return ONLY the code — no markdown fences, no explanation. "
    "Import all necessary libraries at the top of your code block."
)

def demo_intent_to_code():
    print("\n" + "═"*65)
    print("DEMO B-1: INTENT → CODE  (Simulating GitHub Copilot)")
    print("═"*65)

    results = []
    for intent in INTENTS:
        print(f"\n{'─'*60}")
        print(f"📝 Intent:\n{intent}")
        code, elapsed = ask(COPILOT_SYSTEM, intent, max_tokens=500)
        lines = len([l for l in code.split("\n") if l.strip()])
        print(f"\n⚡ Generated in {elapsed}s  ({lines} non-empty lines):")
        print(code[:400] + ("..." if len(code) > 400 else ""))
        results.append({"intent": intent[:60], "lines": lines, "seconds": elapsed})

    print("\n\n📊 Productivity Summary:")
    print(f"{'Intent':<62} {'Lines':>6} {'Time':>7}")
    print("─"*78)
    for r in results:
        print(f"{r['intent']:<62} {r['lines']:>6} {r['seconds']:>6.1f}s")

    avg_time = sum(r["seconds"] for r in results) / len(results)
    avg_lines = sum(r["lines"] for r in results) / len(results)
    print(f"\n  Average: {avg_lines:.0f} lines in {avg_time:.1f}s per task")
    print(f"  Manual estimate: ~8-15 minutes per task")
    print(f"  Speedup factor: ~{8*60/avg_time:.0f}×")


# ---------------------------------------------------------------------------
# 2. /EXPLAIN  (simulates VS Code Copilot Chat)
# ---------------------------------------------------------------------------
COMPLEX_CODE = '''
def process_churn_features(df, target_col="churned", date_cols=None, 
                            lag_windows=[7, 14, 30], top_k_corr=10):
    date_cols = date_cols or []
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        ref = df[col].max()
        df[f"days_since_{col}"] = (ref - df[col]).dt.days
    df.drop(columns=date_cols, inplace=True)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    for col in num_cols:
        for w in lag_windows:
            df[f"{col}_roll{w}_mean"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_roll{w}_std"]  = df[col].rolling(w, min_periods=1).std().fillna(0)
    corr = df[num_cols + [target_col]].corr()[target_col].drop(target_col)
    top_features = corr.abs().nlargest(top_k_corr).index.tolist()
    return df[[target_col] + top_features + [c for c in df.columns if "roll" in c]]
'''

def demo_explain():
    print("\n\n" + "═"*65)
    print("DEMO B-2: /EXPLAIN  (Simulating Copilot Chat)")
    print("═"*65)

    explain_system = (
        "You are GitHub Copilot Chat. Explain code clearly for a mid-level data scientist. "
        "Cover: what it does, inputs/outputs, any gotchas or side effects."
    )
    explanation, elapsed = ask(explain_system,
        f"Explain this Python function:\n```python{COMPLEX_CODE}```", max_tokens=500)
    print(f"\nCode: process_churn_features() [{len(COMPLEX_CODE.split(chr(10)))} lines]")
    print(f"⚡ Explained in {elapsed}s\n")
    print(textwrap.fill(explanation, 70))


# ---------------------------------------------------------------------------
# 3. /TESTS  (simulates Copilot test generation)
# ---------------------------------------------------------------------------
def demo_generate_tests():
    print("\n\n" + "═"*65)
    print("DEMO B-3: /TESTS  (Auto Test Generation)")
    print("═"*65)

    test_system = (
        "You are GitHub Copilot. Generate comprehensive pytest tests for the given function. "
        "Include: happy path, edge cases (empty df, all-null column, wrong dtype), "
        "and at least one parametrize case. Return only executable Python code."
    )
    tests, elapsed = ask(test_system,
        f"Write pytest tests for:\n```python{COMPLEX_CODE}```", max_tokens=700)
    test_lines = len([l for l in tests.split("\n") if l.strip()])
    print(f"\n⚡ Generated {test_lines} lines of tests in {elapsed}s\n")
    print(tests[:600] + ("..." if len(tests) > 600 else ""))


# ---------------------------------------------------------------------------
# 4. /FIX  (simulates Copilot bug detection)
# ---------------------------------------------------------------------------
BUGGY_CODE = '''
def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = y_pred_proba > threshold
    tp = sum(y_true[y_pred == 1] == 1)
    fp = sum(y_true[y_pred == 0] == 1)   # BUG 1: wrong mask
    tn = sum(y_true[y_pred == 0] == 0)
    fn = sum(y_true[y_pred == 1] == 0)   # BUG 2: wrong mask
    precision = tp / (tp + fn)            # BUG 3: wrong denominator
    recall    = tp / (tp + fp)            # BUG 4: wrong denominator
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}
'''

def demo_fix():
    print("\n\n" + "═"*65)
    print("DEMO B-4: /FIX  (AI Bug Detection & Correction)")
    print("═"*65)

    fix_system = (
        "You are GitHub Copilot. Find all bugs in the code. "
        "List each bug with: line, description, and corrected code. "
        "Then provide the fully corrected function."
    )
    fixes, elapsed = ask(fix_system,
        f"Find and fix bugs in:\n```python{BUGGY_CODE}```", max_tokens=700)
    print(f"\nBuggy code: compute_metrics() — contains 4 deliberate bugs")
    print(f"⚡ Bugs found and fixed in {elapsed}s\n")
    print(fixes)


# ---------------------------------------------------------------------------
# 5. Azure ML SDK generation (Azure-specific Copilot use case)
# ---------------------------------------------------------------------------
def demo_azure_sdk_gen():
    print("\n\n" + "═"*65)
    print("DEMO B-5: Azure ML SDK v2 Code Generation")
    print("═"*65)

    intent = """
    Create an Azure ML SDK v2 command job that:
    - Runs train.py with arguments: --learning_rate 0.05 --max_depth 5
    - Uses a Standard_DS3_v2 compute cluster named 'cpu-cluster'
    - Logs to an experiment called 'churn-lgbm'
    - Uses environment: AzureML-sklearn-1.0-ubuntu20.04-py38-cpu
    - Output model to a registered asset called 'churn-model'
    """

    azure_system = (
        "You are a GitHub Copilot expert in Azure ML SDK v2 (azure-ai-ml package). "
        "Generate production-ready SDK v2 code. Include MLClient authentication, "
        "job definition, and job submission. Return only Python code."
    )
    code, elapsed = ask(azure_system, intent, max_tokens=600)
    print(f"\nIntent: Create Azure ML command job with specified parameters")
    print(f"⚡ Azure ML SDK v2 code generated in {elapsed}s\n")
    print(code[:700] + ("..." if len(code) > 700 else ""))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print("█" * 65)
    print("  DEMO B: IDE COPILOT PRODUCTIVITY BENCHMARK")
    print("█" * 65)

    demo_intent_to_code()
    demo_explain()
    demo_generate_tests()
    demo_fix()
    demo_azure_sdk_gen()

    print("\n\n" + "═"*65)
    print("KEY INSIGHT: All 5 IDE tasks above completed via API calls.")
    print("GitHub Copilot does this inline as you type — same model,")
    print("better UX. The productivity gain is real and measurable.")
    print("═"*65)


if __name__ == "__main__":
    run()
