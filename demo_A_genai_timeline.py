"""
=============================================================================
DEMO A: GenAI Timeline Explorer
=============================================================================
Session: "LLMs — Yes or No with Data Science?"
Topic:   Interactive exploration of GenAI milestones + GPT-4o milestone Q&A

Shows:
  1. Plotly interactive timeline of GenAI milestones coloured by era
  2. GPT-4o contextual analysis of each era's impact on data science
  3. "Where were you in 2022?" — personalised milestone reflection via LLM
  4. Azure milestone overlay with ds impact scoring

Prerequisites:
    /opt/anaconda3/bin/python -m pip install plotly openai pandas python-dotenv
Local env.:
    base (3.13.9)
/opt/anaconda3/bin/python -c "import pandas; print(pandas.__version__)"
conda install pandas
=============================================================================
"""

import os, json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OAI_ENDPOINT"],
    api_key=os.environ["AZURE_OAI_KEY"],
    api_version="2024-02-01",
)
MODEL = os.environ.get("AZURE_OAI_DEPLOY", "gpt-4o")


# ---------------------------------------------------------------------------
# 1. Milestone dataset
# ---------------------------------------------------------------------------
MILESTONES = [
    # Foundation Era
    {"year": 2017.5, "event": "Transformer Paper",    "era": "Foundation", "impact": 5,  "azure": False, "detail": "Attention is All You Need — Vaswani et al. Every modern LLM is built on this."},
    {"year": 2018.0, "event": "BERT",                 "era": "Foundation", "impact": 6,  "azure": False, "detail": "Bidirectional pre-training. Made NLP fine-tuning practical for most companies."},
    {"year": 2018.5, "event": "GPT-1",                "era": "Foundation", "impact": 4,  "azure": False, "detail": "OpenAI's first generative pre-trained transformer. 117M parameters."},
    {"year": 2019.0, "event": "GPT-2",                "era": "Foundation", "impact": 5,  "azure": False, "detail": "Too dangerous to release (OpenAI). Showed generative text quality breakthrough."},
    {"year": 2019.5, "event": "XLNet / RoBERTa",      "era": "Foundation", "impact": 4,  "azure": False, "detail": "Improved BERT training. State-of-the-art on GLUE benchmarks."},
    # Scaling Era
    {"year": 2020.3, "event": "GPT-3 (175B)",         "era": "Scaling",    "impact": 9,  "azure": False, "detail": "Few-shot learning at scale. First time prompting replaced fine-tuning for many tasks."},
    {"year": 2021.0, "event": "DALL-E / CLIP",        "era": "Scaling",    "impact": 6,  "azure": False, "detail": "Multimodal foundation models. Text-to-image changed creative workflows."},
    {"year": 2021.5, "event": "Codex / GitHub Copilot","era": "Scaling",   "impact": 8,  "azure": True,  "detail": "Code generation became real. GitHub Copilot launched as first mass-market LLM product."},
    {"year": 2022.0, "event": "Azure OpenAI Preview",  "era": "Scaling",   "impact": 7,  "azure": True,  "detail": "Azure made GPT-3 available in a private preview. Enterprise data science changed."},
    {"year": 2022.7, "event": "ChatGPT   ",            "era": "RLHF",      "impact": 10, "azure": False, "detail": "100M users in 60 days. RLHF made LLMs reliably instruction-following. The inflection point."},
    {"year": 2022.9, "event": "Stable Diffusion",      "era": "Scaling",   "impact": 6,  "azure": False, "detail": "Open-source image generation. Democratised generative art and data augmentation."},
    # RLHF Era
    {"year": 2023.1, "event": "GPT-4",                "era": "RLHF",      "impact": 9,  "azure": True,  "detail": "Multimodal (vision). Available on Azure OpenAI. Became DS coding assistant standard."},
    {"year": 2023.3, "event": "Azure AI Studio",       "era": "RLHF",      "impact": 7,  "azure": True,  "detail": "Unified portal for Azure OpenAI + ML. Prompt Flow introduced for LLMOps."},
    {"year": 2023.5, "event": "Llama 2 (Meta)",       "era": "RLHF",      "impact": 7,  "azure": False, "detail": "Open-source LLM. Available in Azure Model Catalog. Changed enterprise self-hosting calculus."},
    {"year": 2023.7, "event": "Claude 2",             "era": "RLHF",      "impact": 6,  "azure": False, "detail": "200K context window. Long document analysis became viable for the first time."},
    {"year": 2023.9, "event": "Mistral 7B",           "era": "RLHF",      "impact": 6,  "azure": False, "detail": "Efficient open model. Showed small models could compete with large ones on many tasks."},
    # Agentic Era
    {"year": 2024.1, "event": "Azure AI Foundry GA",  "era": "Agentic",   "impact": 8,  "azure": True,  "detail": "Unified ML + GenAI platform. Model catalog with 1800+ models. LLMOps at enterprise scale."},
    {"year": 2024.2, "event": "GPT-4o",               "era": "Agentic",   "impact": 8,  "azure": True,  "detail": "Omni model — text/audio/vision in one. Real-time voice. Available on Azure OpenAI."},
    {"year": 2024.4, "event": "Claude 3 / Gemini 1.5","era": "Agentic",   "impact": 7,  "azure": False, "detail": "1M+ token context. Entire codebases and datasets in single context window."},
    {"year": 2024.7, "event": "Llama 3.1 405B",      "era": "Agentic",   "impact": 7,  "azure": True,  "detail": "Open-source model matching GPT-4. Available in Azure Model Catalog for self-hosted use."},
    {"year": 2024.9, "event": "Azure Agent Service",  "era": "Agentic",   "impact": 8,  "azure": True,  "detail": "Managed agent orchestration on Azure. Multi-agent frameworks with AutoGen."},
    {"year": 2025.1, "event": "o3 / o4-mini",        "era": "Agentic",   "impact": 9,  "azure": True,  "detail": "Reasoning models. AIME/math/code competition-level performance. Chain-of-thought native."},
    {"year": 2025.3, "event": "DeepSeek R1",          "era": "Agentic",   "impact": 7,  "azure": False, "detail": "Open-source reasoning model. Shocked the market. Cost-efficient alternative to o1."},
]

df = pd.DataFrame(MILESTONES)

ERA_COLORS = {
    "Foundation": "#0078D4",
    "Scaling":    "#1E8449",
    "RLHF":       "#8E44AD",
    "Agentic":    "#D13438",
}


# ---------------------------------------------------------------------------
# 2. Build interactive Plotly timeline
# ---------------------------------------------------------------------------
def build_timeline() -> go.Figure:
    fig = go.Figure()

    # Era background bands
    era_ranges = {"Foundation": (2017, 2020), "Scaling": (2020, 2022.6),
                  "RLHF": (2022.6, 2024), "Agentic": (2024, 2025.6)}

    for era, (x0, x1) in era_ranges.items():
        fig.add_vrect(x0=x0, x1=x1,
                      fillcolor=ERA_COLORS[era], opacity=0.06,
                      annotation_text=f"<b>{era} Era</b>",
                      annotation_position="top left",
                      annotation_font_color=ERA_COLORS[era])

    # Plot milestones per era
    for era, color in ERA_COLORS.items():
        sub = df[df["era"] == era]
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["impact"],
            mode="markers+text",
            name=era,
            marker=dict(size=sub["impact"] * 4, color=color,
                        line=dict(width=2, color="white"), opacity=0.85),
            text=sub["event"], textposition="top center",
            hovertemplate="<b>%{text}</b><br>Year: %{x:.1f}<br>Impact: %{y}/10<br>%{customdata}",
            customdata=sub["detail"],
        ))

    # Azure milestones — star markers
    azure = df[df["azure"]]
    fig.add_trace(go.Scatter(
        x=azure["year"], y=azure["impact"] + 0.5,
        mode="markers", name="☁️ Azure milestone",
        marker=dict(symbol="star", size=14, color="#FFB900",
                    line=dict(width=1, color="white")),
        hovertemplate="<b>Azure:</b> %{customdata}",
        customdata=azure["event"],
    ))

    fig.update_layout(
        title=dict(text="<b>GenAI Milestone Timeline (2017–2025)</b><br>"
                        "<sub>Marker size = DS impact score | ⭐ = Azure milestone</sub>",
                   font_size=18),
        xaxis=dict(title="Year", tickformat=".0f", dtick=1, gridcolor="#eee"),
        yaxis=dict(title="Data Science Impact (1–10)", range=[2, 12], gridcolor="#eee"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.15),
        height=560, font=dict(family="Calibri", size=12),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. GPT-4o era analysis
# ---------------------------------------------------------------------------
def analyse_era(era: str) -> str:
    milestones_in_era = [m["event"] for m in MILESTONES if m["era"] == era]
    prompt = f"""
You are a data science historian. Analyse the "{era} Era" of GenAI (milestones: {', '.join(milestones_in_era)}).

Answer concisely:
1. What was the defining characteristic of this era for DATA SCIENTISTS (not ML researchers)?
2. What specific Azure service/capability emerged or changed in this era?
3. What was the ONE task that went from "hard" to "easy" during this era?
4. What skill became newly important for DS practitioners?

Format as 4 bullet points."""
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=400
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# 4. "Milestone impact assessment" — personalised Q&A
# ---------------------------------------------------------------------------
def milestone_qa(question: str) -> str:
    context = "\n".join([
        f"- {m['year']:.0f}: {m['event']} (impact {m['impact']}/10) — {m['detail']}"
        for m in MILESTONES
    ])
    prompt = f"""You are an expert on the history of GenAI and data science.
Use the following milestone database to answer the question.

Milestones:
{context}

Question: {question}

Answer in 3-5 sentences, referencing specific milestones by name."""
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=350
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def run():
    print("  DEMO A: GenAI TIMELINE EXPLORER")
 

    # Build and save timeline
    fig = build_timeline()
    fig.write_html("genai_timeline.html")
    print("\n Saved: genai_timeline.html — open in browser for interactive view")
    fig.write_image("genai_timeline.png", width=1200, height=600, scale=1.5)
    print(" Saved: genai_timeline.png")

    # Era analyses
    print("\n--- ERA-BY-ERA ANALYSIS (Azure OpenAI GPT-4o) ---\n")
    for era in ERA_COLORS:
        print(f"\n{'='*55}")
        print(f" {era.upper()} ERA")
        print('='*55)
        analysis = analyse_era(era)
        print(analysis)

    # Q&A demo
    sample_questions = [
        "What was the single most impactful milestone for data scientists who primarily worked on tabular data?",
        "When did Azure become a serious competitor to direct OpenAI API access, and why?",
        "How did the RLHF era change the way data scientists evaluate model quality?",
    ]
    print("\n\n--- MILESTONE Q&A ---")
    for q in sample_questions:
        print(f"\n {q}")
        print(f" {milestone_qa(q)}")

    # Export summary CSV
    df.to_csv("genai_milestones.csv", index=False)
    print("\n Saved: genai_milestones.csv")


if __name__ == "__main__":
    run()
