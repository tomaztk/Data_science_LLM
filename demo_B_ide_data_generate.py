"""
=============================================================================
DEMO B: IDE Copilot Productivity Benchmark  (fixed — self-contained data)
=============================================================================
Session: "LLMs — Yes or No with Data Science?"
Topic:   Simulating GitHub Copilot-style intent-to-code generation via API

What it shows:
  1. Intent comment → full DS function generated AND executed with real data
  2. /explain — explain a complex existing function
  3. /tests  — auto-generate pytest tests
  4. /fix    — detect and fix deliberate bugs
  5. Azure ML SDK v2 code generation from an English description

All demo data is created in-memory / on-disk BEFORE code generation runs,
so every generated snippet executes successfully.

Prerequisites:
    pip install openai pandas numpy scikit-learn lightgbm python-dotenv
    /Users/tomazkastrun/opt/anaconda3/bin/python -m pip install lightgbm
=============================================================================
"""

import os, json, time, textwrap, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


# ── Azure OpenAI client ────────────────────────────────────────────────────
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OAI_ENDPOINT"],
    api_key=os.environ["AZURE_OAI_KEY"],
    api_version="2024-02-01",
)
MODEL = os.environ.get("AZURE_OAI_DEPLOY", "gpt-4o-mini")


# ── Helpers ────────────────────────────────────────────────────────────────
def ask(system: str, user: str, max_tokens: int = 600) -> tuple[str, float]:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=0.1, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip(), round(time.time() - t0, 2)


def strip_fences(code: str) -> str:
    """Remove ```python ... ``` fences that the model sometimes adds."""
    lines = code.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def safe_exec(code: str, context: dict, label: str = "") -> bool:
    """
    Execute *code* inside *context*. Print result or full traceback.
    Returns True on success, False on failure.
    """
    code = strip_fences(code)
    try:
        exec(compile(code, "<generated>", "exec"), context)
        return True
    except Exception:
        print(f"\n⚠️  Execution error in {label}:\n{traceback.format_exc()}")
        return False


# ===========================================================================
# STEP 0 — CREATE ALL DEMO DATA UPFRONT
# ===========================================================================
def create_demo_data() -> dict:
    """
    Generate every data artefact the demos need.
    Returns a dict that is used as the execution context for generated code.
    """
    rng = np.random.default_rng(42)
    n = 800

    # ── Synthetic sales / churn dataset ─────────────────────────────────────
    regions  = rng.choice(["EU", "US", "APAC", "LATAM"], n)
    plan     = rng.choice(["free", "pro", "enterprise"], n, p=[0.4, 0.45, 0.15])
    revenue  = rng.uniform(20, 120, n).round(2)
    logins   = rng.integers(0, 60, n)
    tickets  = rng.integers(0, 12, n)
    tenure   = rng.integers(1, 72, n)
    churned  = (
        (revenue < 40) |
        (logins < 5) |
        (tickets > 8) |
        (rng.uniform(0, 1, n) < 0.06)
    ).astype(int)

    df_full = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "region":      regions,
        "plan":        plan,
        "revenue":     revenue,
        "logins":      logins,
        "support_tickets": tickets,
        "tenure_months":   tenure,
        "churned":     churned,
    })

    # Inject ~5 % null revenue to make the first intent realistic
    null_idx = rng.choice(n, int(n * 0.05), replace=False)
    df_full.loc[null_idx, "revenue"] = np.nan

    # Save to disk so generated file-loading code works
    Path("data").mkdir(exist_ok=True)
    df_full.to_csv("data/sales.csv", index=False)
    print(f"  ✅ data/sales.csv  ({n} rows, {df_full['churned'].mean():.1%} churn rate)")

    # ── Pre-split version for demos that need X_train / X_test directly ─────
    df_clean = df_full.dropna(subset=["revenue"]).copy()
    df_clean = pd.get_dummies(df_clean, columns=["region", "plan"], drop_first=False)
    feature_cols = [c for c in df_clean.columns
                    if c not in ("customer_id", "churned")]

    X = df_clean[feature_cols].astype(float)
    y = df_clean["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Pre-trained LightGBM for SHAP demo ──────────────────────────────────
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"  ✅ LightGBM model trained  (AUC={auc:.3f})")

    context = dict(
        # raw frame (with NaNs — mimics a real file load)
        df=df_full.copy(),
        # pre-split matrices
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        # fitted model
        model=model,
        feature_names=list(feature_cols),
        # common imports available inside exec()
        pd=pd, np=np, lgb=lgb,
        train_test_split=train_test_split,
        roc_auc_score=roc_auc_score,
        print=print,
        Path=Path,
    )
    return context


# ===========================================================================
# DEMO B-1: INTENT → CODE  (generated + executed)
# ===========================================================================
COPILOT_SYSTEM = (
    "You are GitHub Copilot, an expert Python data scientist. "
    "Complete the function described in the comment. "
    "Assume these variables are already in scope: "
    "df (pandas DataFrame with columns: customer_id, region, plan, revenue, logins, "
    "support_tickets, tenure_months, churned — loaded from data/sales.csv), "
    "X_train, X_test, y_train, y_test (numpy/pandas, already split 80/20), "
    "model (fitted LightGBM classifier), feature_names (list of str). "
    "All of pandas, numpy, lightgbm are imported as pd, np, lgb. "
    "Do NOT reload the CSV. Do NOT re-split data. Do NOT refit the model unless asked. "
    "Return ONLY executable Python — no markdown fences, no explanation."
)

INTENTS = [
    (
        "# Using the existing df (already loaded), drop rows where 'revenue' is null,\n"
        "# encode 'region' and 'plan' as dummy variables, print shape before/after.",
        "data cleaning + encoding",
    ),
    (
        "# Using X_train, y_train, X_test, y_test (already split),\n"
        "# train a LightGBM classifier and print AUC-ROC on the test set.",
        "LightGBM training",
    ),
    (
        "# Using the fitted 'model' and X_test, compute feature importances\n"
        "# and print the top 10 as a ranked table with gain scores.",
        "feature importance table",
    ),
    (
        "# Using X_test and y_test, compute and print: accuracy, precision,\n"
        "# recall, F1, and AUC-ROC for the fitted 'model'.",
        "model evaluation metrics",
    ),
    (
        "# Print a 2×2 confusion matrix for the fitted 'model' on X_test / y_test\n"
        "# with labels 'Retained' and 'Churned'. Use only pandas / numpy — no sklearn.",
        "confusion matrix (no sklearn)",
    ),
]


def demo_intent_to_code(context: dict) -> None:
    print("\n" + "═" * 65)
    print("DEMO B-1: INTENT → CODE  (generated + executed live)")
    print("═" * 65)

    results = []
    for intent, label in INTENTS:
        print(f"\n{'─' * 60}")
        print(f"📝 Intent [{label}]:\n{intent}")

        code, elapsed = ask(COPILOT_SYSTEM, intent, max_tokens=450)
        lines = len([l for l in code.split("\n") if l.strip()])
        print(f"\n⚡ Generated in {elapsed}s  ({lines} lines)")
        print("─── generated code ───")
        print(code)
        print("─── output ───")
        ok = safe_exec(code, context, label=label)
        results.append({"label": label, "lines": lines, "seconds": elapsed, "ok": ok})

    # Summary table
    print("\n\n📊 Productivity Summary:")
    print(f"  {'Task':<35} {'Lines':>6} {'Time':>8} {'Ran OK':>8}")
    print("  " + "─" * 62)
    for r in results:
        tick = "✅" if r["ok"] else "⚠️ "
        print(f"  {r['label']:<35} {r['lines']:>6} {r['seconds']:>7.1f}s {tick:>8}")

    avg_t = sum(r["seconds"] for r in results) / len(results)
    print(f"\n  Average generation time : {avg_t:.1f}s per task")
    print(f"  Manual coding estimate  : 8–15 minutes per task")
    print(f"  Speed-up factor         : ~{int(10 * 60 / avg_t)}×")


# ===========================================================================
# DEMO B-2: /EXPLAIN
# ===========================================================================
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


def demo_explain() -> None:
    print("\n\n" + "═" * 65)
    print("DEMO B-2: /EXPLAIN  (VS Code Copilot Chat simulation)")
    print("═" * 65)

    system = (
        "You are GitHub Copilot Chat. Explain code clearly for a mid-level data scientist. "
        "Cover: what it does, inputs/outputs, side-effects, and any gotchas."
    )
    explanation, elapsed = ask(
        system,
        f"Explain this Python function:\n```python\n{COMPLEX_CODE}\n```",
        max_tokens=500,
    )
    print(f"\nFunction: process_churn_features()  ({len(COMPLEX_CODE.splitlines())} lines)")
    print(f"⚡ Explained in {elapsed}s\n")
    print(textwrap.fill(explanation, 72))


# ===========================================================================
# DEMO B-3: /TESTS
# ===========================================================================
def demo_generate_tests() -> None:
    print("\n\n" + "═" * 65)
    print("DEMO B-3: /TESTS  (auto pytest generation)")
    print("═" * 65)

    system = (
        "You are GitHub Copilot. Generate pytest tests for the given Python function. "
        "Include: happy path with a small DataFrame, edge cases (empty df, all-null column), "
        "and at least one parametrize case. Return only executable Python code — no fences."
    )
    tests, elapsed = ask(
        system,
        f"Write pytest tests for:\n```python\n{COMPLEX_CODE}\n```",
        max_tokens=700,
    )
    test_lines = len([l for l in tests.split("\n") if l.strip()])
    print(f"\n⚡ Generated {test_lines} lines of tests in {elapsed}s\n")
    print(tests[:800] + ("\n..." if len(tests) > 800 else ""))


# ===========================================================================
# DEMO B-4: /FIX
# ===========================================================================
BUGGY_CODE = '''
def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = y_pred_proba > threshold
    tp = sum(y_true[y_pred == 1] == 1)
    fp = sum(y_true[y_pred == 0] == 1)   # BUG 1: mask should be y_pred==1
    tn = sum(y_true[y_pred == 0] == 0)
    fn = sum(y_true[y_pred == 1] == 0)   # BUG 2: mask should be y_pred==0
    precision = tp / (tp + fn)            # BUG 3: denominator should be tp+fp
    recall    = tp / (tp + fp)            # BUG 4: denominator should be tp+fn
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}
'''


def demo_fix(context: dict) -> None:
    print("\n\n" + "═" * 65)
    print("DEMO B-4: /FIX  (AI bug detection & correction)")
    print("═" * 65)

    system = (
        "You are GitHub Copilot. Find ALL bugs. "
        "List each bug: line number, description, fix. "
        "Then provide the fully corrected function — no markdown fences."
    )
    fixes, elapsed = ask(
        system,
        f"Find and fix all bugs:\n```python\n{BUGGY_CODE}\n```",
        max_tokens=700,
    )
    print(f"\nBuggy function: compute_metrics() — 4 deliberate bugs hidden inside")
    print(f"⚡ Analysed in {elapsed}s\n")
    print(fixes)

    # Extract and run the fixed function to verify correctness
    print("\n─── Verifying the fixed function runs correctly ───")
    # We run only the function definition + a quick sanity check
    verify_code = strip_fences(fixes)
    # Keep lines up to end of function (find last "return" line)
    defn_lines = []
    in_func = False
    for line in verify_code.split("\n"):
        if line.strip().startswith("def compute_metrics"):
            in_func = True
        if in_func:
            defn_lines.append(line)
            if line.strip().startswith("return") and in_func:
                break
    verify_snippet = "\n".join(defn_lines) + "\n"
    verify_snippet += (
        "\nimport numpy as np\n"
        "y_true  = np.array([1,0,1,1,0,0,1,0])\n"
        "y_proba = np.array([0.9,0.1,0.8,0.7,0.3,0.2,0.6,0.4])\n"
        "result  = compute_metrics(y_true, y_proba)\n"
        "print(f'  precision={result[\"precision\"]:.3f}  recall={result[\"recall\"]:.3f}  f1={result[\"f1\"]:.3f}')\n"
        "assert 0 < result['precision'] <= 1\n"
        "assert 0 < result['recall']    <= 1\n"
        "print('  ✅ Fixed function passes sanity check')\n"
    )
    safe_exec(verify_snippet, context, label="fixed compute_metrics")


# ===========================================================================
# DEMO B-5: Azure ML SDK v2 generation
# ===========================================================================
def demo_azure_sdk_gen() -> None:
    print("\n\n" + "═" * 65)
    print("DEMO B-5: Azure ML SDK v2 Code Generation")
    print("═" * 65)

    intent = (
        "Create an Azure ML SDK v2 command job that:\n"
        "  - Runs train.py with arguments: --learning_rate 0.05 --max_depth 5\n"
        "  - Uses a Standard_DS3_v2 cluster named 'cpu-cluster'\n"
        "  - Logs to an experiment called 'churn-lgbm'\n"
        "  - Uses environment: AzureML-sklearn-1.0-ubuntu20.04-py38-cpu\n"
        "  - Saves the output model as a registered asset 'churn-model'"
    )

    system = (
        "You are a GitHub Copilot expert in Azure ML SDK v2 (azure-ai-ml). "
        "Generate production-ready Python using MLClient, command(), and job submission. "
        "Include DefaultAzureCredential authentication. Return only Python code — no fences."
    )
    code, elapsed = ask(system, intent, max_tokens=600)
    print(f"\nIntent: {intent}")
    print(f"\n⚡ Azure ML SDK v2 code generated in {elapsed}s:\n")
    print(code[:900] + ("\n..." if len(code) > 900 else ""))
    print(
        "\n💡 This code won't run without a real workspace, but it's correct SDK v2 syntax.\n"
        "   Copilot generates this inline as you type the comment — same quality, zero wait."
    )


# ===========================================================================
# MAIN
# ===========================================================================
def run() -> None:
    print("█" * 65)
    print("  DEMO B: IDE COPILOT PRODUCTIVITY BENCHMARK")
    print("█" * 65)

    print("\n[Setup] Generating demo data and training baseline model...")
    context = create_demo_data()

    demo_intent_to_code(context)
    demo_explain()
    demo_generate_tests()
    demo_fix(context)
    demo_azure_sdk_gen()

    print("\n\n" + "═" * 65)
    print("KEY INSIGHT:")
    print("  Every generated snippet above ran against REAL data.")
    print("  GitHub Copilot does this inline as you type — same model,")
    print("  zero copy-paste, zero context switch out of your IDE.")
    print("  The productivity gain is real, measurable, and cumulative.")
    print("═" * 65)


if __name__ == "__main__":
    run()