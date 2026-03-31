"""
=============================================================================
DEMO D: LLM-Powered Data Visualisation
=============================================================================
Session: "LLMs — Yes or No with Data Science?"
Topic:   NL → chart code, chart → narrative, multimodal chart analysis

Shows:
  1. Natural language → matplotlib code (generated + executed)
  2. Natural language → Plotly interactive chart code
  3. Chart image → stakeholder narrative (GPT-4o vision)
  4. Automated chart narration pipeline for dashboards
  5. Power BI Copilot equivalent: DAX measure generation from description

Prerequisites:
    pip install openai matplotlib plotly pandas numpy kaleido python-dotenv
=============================================================================
"""

import os
import json
import base64
import textwrap
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.graph_objects as go
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




# ---------------------------------------------------------------------------
# 0. Generate sample business data
# ---------------------------------------------------------------------------
np.random.seed(42)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
revenue = np.array([120, 135, 148, 142, 155, 168, 172, 185, 178, 192, 205, 218]) * 1000
churn_rate = np.array([5.2, 4.8, 4.5, 5.1, 4.2, 3.9, 3.7, 3.5, 3.8, 3.2, 2.9, 2.7])
cohort_sizes = {"2022 Q1": 1200, "2022 Q2": 980, "2022 Q3": 1450, "2022 Q4": 1100}
retention = {"2022 Q1": [100, 82, 71, 65, 59, 54], "2022 Q2": [100, 79, 68, 61, 55, 51],
             "2022 Q3": [100, 85, 76, 69, 64, 59], "2022 Q4": [100, 80, 70, 63, 57, 53]}


def ask_for_code(prompt: str, style: str = "matplotlib") -> str:
    system = (
        f"You are a data visualisation expert. Write clean, professional Python {style} code. "
        "Use actual data variables already defined in scope (not dummy data). "
        "Return ONLY executable Python code — no markdown fences, no explanation, no imports "
        "unless strictly necessary beyond what's already imported."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=700,
    )
    return resp.choices[0].message.content.strip()


def safe_exec(code: str, local_vars: dict) -> str:
    """Execute generated code safely, return error string if it fails."""
    try:
        exec(code, {**local_vars, "plt": plt, "np": np, "pd": pd, "go": go,
                    "mtick": mtick, "__builtins__": __builtins__})
        return "OK"
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# 1. NL → MATPLOTLIB CODE
# ---------------------------------------------------------------------------
def demo_nl_to_matplotlib():
    print("\n" + "═"*65)
    print("DEMO D-1: NATURAL LANGUAGE → MATPLOTLIB CODE")
    print("═"*65)

    requests = [
        {
            "request": "Create a two-panel figure: left panel shows monthly revenue as bars with a trend line overlay; right panel shows churn rate as a line with the y-axis as percentage. Use a clean, board-presentation style with navy and teal colours.",
            "fname": "chart_revenue_churn.png",
        },
        {
            "request": "Plot a cohort retention heatmap. Rows = cohorts, columns = months 0-5, values = retention %. Use a blue-to-red diverging colormap where 100% is blue and lower is more red. Add text annotations for each cell.",
            "fname": "chart_cohort_heatmap.png",
        },
    ]

    local_vars = {"MONTHS": MONTHS, "revenue": revenue, "churn_rate": churn_rate,
                  "retention": retention, "cohort_sizes": cohort_sizes}

    for req in requests:
        print(f"\n Request: {req['request'][:80]}...")
        code = ask_for_code(req["request"] + f"\n\nData available: MONTHS={MONTHS}, revenue={list(revenue[:3])}... (12 items), churn_rate={list(churn_rate[:3])}... (12 items), retention=dict of cohorts with 6 months each.")
        print(f"\n Generated code ({len(code.split(chr(10)))} lines):")
        print(code[:400] + ("..." if len(code) > 400 else ""))

        plt.figure(figsize=(14, 5))
        result = safe_exec(code, local_vars)
        if result == "OK":
            plt.savefig(req["fname"], dpi=120, bbox_inches="tight")
            plt.close()
            print(f" Saved: {req['fname']}")
        else:
            plt.close()
            print(f"  {result} — (fix prompt and retry in production)")


# ---------------------------------------------------------------------------
# 2. NL → PLOTLY INTERACTIVE CHART
# ---------------------------------------------------------------------------
def demo_nl_to_plotly():
    print("\n\n" + "═"*65)
    print("DEMO D-2: NATURAL LANGUAGE → PLOTLY INTERACTIVE CHART")
    print("═"*65)

    request = (
        "Create a Plotly figure showing monthly revenue as bars and churn rate as a line "
        "on a secondary y-axis. Add hover tooltips with exact values. "
        "Use #0078D4 for bars, #D13438 for the churn line. Title: 'Revenue vs Churn Rate 2024'. "
        "Export as HTML to 'interactive_chart.html' using fig.write_html()."
    )
    print(f" Request: {request}")

    code = ask_for_code(request + f"\nData: MONTHS={MONTHS}, revenue (array, USD)={list(revenue)}, churn_rate (%, array)={list(churn_rate)}", style="Plotly")
    print(f"\n Generated code:\n{code[:500]}...")

    local_vars = {"MONTHS": MONTHS, "revenue": revenue, "churn_rate": churn_rate}
    result = safe_exec(code, local_vars)
    if result == "OK":
        print(" Saved: interactive_chart.html")
    else:
        print(f"  {result}")


# ---------------------------------------------------------------------------
# 3. CHART → NARRATIVE  (GPT-4o Vision)
# ---------------------------------------------------------------------------
def demo_chart_to_narrative():
    print("\n\n" + "═"*65)
    print("DEMO D-3: CHART → NARRATIVE  (GPT-4o Multimodal Vision)")
    print("═"*65)

    # Create a chart to analyse
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(MONTHS, revenue / 1000, color="#0078D4", alpha=0.8, label="Revenue ($K)")
    ax2 = ax.twinx()
    ax2.plot(MONTHS, churn_rate, "o-", color="#D13438", linewidth=2.5, label="Churn Rate (%)")
    ax.set_ylabel("Monthly Revenue ($K)", color="#0078D4")
    ax2.set_ylabel("Churn Rate (%)", color="#D13438")
    ax.set_title("Revenue & Churn Rate — 2024")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("chart_for_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Encode image as base64
    with open("chart_for_analysis.png", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    audiences = [
        ("CFO", "3 bullet points, focus on financial impact and trend direction, in USD terms"),
        ("Sales team", "actionable insights about which months had best/worst churn to inform retention campaigns"),
        ("Board", "one-paragraph executive summary with key risk and opportunity"),
    ]

    for audience, instruction in audiences:
        print(f"\n Narrating chart for: {audience.upper()}")
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": f"Analyse this business chart for a {audience}. Provide {instruction}. Be specific with numbers."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]}],
            max_tokens=300, temperature=0.2,
        )
        narrative = resp.choices[0].message.content
        print(textwrap.fill(narrative, 70, initial_indent="  ", subsequent_indent="  "))


# ---------------------------------------------------------------------------
# 4. AUTOMATED NARRATION PIPELINE (dashboard automation)
# ---------------------------------------------------------------------------
def demo_automated_narration():
    print("\n\n" + "═"*65)
    print("DEMO D-4: AUTOMATED NARRATION PIPELINE")
    print("═"*65)
    print("(Simulates a scheduled pipeline that generates narrative for fresh data)\n")

    # Simulate "fresh" data from a BI system
    report_data = {
        "period": "December 2024",
        "revenue": 218000,
        "revenue_vs_prev": +6.5,
        "churn_rate": 2.7,
        "churn_vs_prev": -0.2,
        "new_customers": 312,
        "churned_customers": 88,
        "avg_order_value": 698,
        "top_region": "APAC",
        "top_region_growth": 12.3,
    }

    prompt = f"""
You are a BI analyst. Generate a monthly performance summary for this data:
{json.dumps(report_data, indent=2)}

Structure:
1. Headline (one bold sentence)
2. Three key metrics with YoY/MoM context
3. One risk and one opportunity
4. Recommended action for next month

Keep it under 200 words. Write for a senior leadership team."""

    narrative = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=350
    ).choices[0].message.content

    print(f"Input data: {json.dumps(report_data, indent=2)}")
    print(f"\n Auto-Generated Narrative:\n{'─'*60}")
    print(narrative)
    print('─'*60)
    print("\n This runs on a schedule — every dashboard refresh generates fresh narrative.")


# ---------------------------------------------------------------------------
# 5. POWER BI COPILOT EQUIVALENT — DAX measure generation
# ---------------------------------------------------------------------------
def demo_dax_generation():
    print("\n\n" + "═"*65)
    print("DEMO D-5: POWER BI COPILOT EQUIVALENT (DAX Generation)")
    print("═"*65)

    dax_schema = """
Tables:
  Sales[OrderDate], Sales[Revenue], Sales[CustomerID], Sales[ProductCategory]
  Customers[CustomerID], Customers[Plan], Customers[Region], Customers[Churned]
  Calendar[Date], Calendar[Year], Calendar[Month], Calendar[Quarter]
"""

    dax_requests = [
        "Running 3-month average revenue",
        "Churn rate for enterprise customers this quarter vs last quarter",
        "Revenue per customer by region, excluding churned customers",
    ]

    dax_system = (
        "You are a Power BI / DAX expert. Write correct, optimised DAX measures. "
        "Use CALCULATE, FILTER, DATESINPERIOD, and time intelligence functions appropriately. "
        "Return only the DAX measure code with a name."
    )

    for req in dax_requests:
        print(f"\n Request: {req}")
        dax = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": dax_system},
                      {"role": "user", "content": f"Schema:\n{dax_schema}\n\nWrite DAX for: {req}"}],
            temperature=0.1, max_tokens=200,
        ).choices[0].message.content
        print(f"Generated DAX:\n{dax}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print("█" * 65)
    print("  DEMO D: LLM-POWERED DATA VISUALISATION")
    print("█" * 65)

    demo_nl_to_matplotlib()
    demo_nl_to_plotly()
    demo_chart_to_narrative()
    demo_automated_narration()
    demo_dax_generation()

    print("\n All visualisation demos complete.")
    print("   Outputs: chart_revenue_churn.png, chart_cohort_heatmap.png,")
    print("            interactive_chart.html, chart_for_analysis.png")


if __name__ == "__main__":
    run()
