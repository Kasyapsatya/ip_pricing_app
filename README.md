# IP Pricing Agent — Streamlit App

ABC Health | 9th IAI Capacity Building Seminar in Health and Care Insurance

A working application wrapping the locked IP semi-Markov pricing model
(`ip_semi_markov_pricing_agent.ipynb`) in a dual-interface Streamlit app:
a deterministic quote calculator in the sidebar, and a conversational
Agno agent ("Priya Nair") in the main canvas that answers follow-up
questions using the exact same three pricing tools, with a real tool-call
trace shown under every answer.

## File structure

```
ip_pricing_app/
├── app.py                  # the Streamlit app itself
├── pricing_tools.py        # all pricing logic, ported verbatim from the notebook
├── build_data.py           # ONE-TIME script: simulates data, saves data/pricing_artifacts.pkl
├── chat_store.py           # SQLite-backed chat history (survives app restarts)
├── requirements.txt
├── .streamlit/
│   └── config.toml         # IAI colour theme
├── assets/
│   └── logo.jpeg           # SSSIA seal
└── data/
    ├── pricing_artifacts.pkl   # pre-built — the app never re-simulates
    └── chat_history.db         # created automatically on first run
```

## Setup

```bash
pip install -r requirements.txt
```

The pricing data is **already generated and included** (`data/pricing_artifacts.pkl`).
You only need to re-run `python build_data.py` if you deliberately change a
constant in `pricing_tools.py` (e.g. `N_LIVES`, `AGE_BASE_INCIDENCE`, etc.)
and want a fresh dataset — the app itself never regenerates data on its own.

## API key

The app looks for a Gemini API key in this order:

1. `st.secrets["GOOGLE_API_KEY"]` — for local use, create `.streamlit/secrets.toml`:
   ```toml
   GOOGLE_API_KEY = "your-key-here"
   ```
   For Streamlit Community Cloud, set this under your app's **Settings → Secrets**.
2. If no secret is found, the sidebar shows a password-style input box to paste
   a key for the current session only (never written to disk).

## Run

```bash
streamlit run app.py
```

## What's deterministic vs. agentic

- **"Calculate Quote" button (sidebar form)** — calls `calculate_premium(...)`
  directly, no LLM involved. Always reproducible, always fast.
- **Chat ("Ask Priya Nair")** — an Agno agent with the same tools
  (`calculate_premium`, `explain_transition`, `explain_occupation`,
  `explain_deferred_option`, `explain_loading`, and the four guardrail
  `check_*` functions) reasons over the conversation and decides which
  tools to call. Every tool call the agent actually makes is shown in an
  expandable **"🔧 Tool trace"** under its answer — extracted from Agno's
  real `RunResponse.messages`, not a summary written after the fact.

## Notes carried over from the notebook

- Guardrails mean the agent will refuse to invent a number for anything
  outside the three tools' tables (a third occupation class, a non-standard
  deferred period, a smoker-status loading, etc.) — see the notebook's
  Section 9/10 ("Break It" / "Guardrail Fix") for a worked example of why
  this matters.
- All pricing data is **illustrative/synthetic**, generated once via
  `build_data.py`, not fitted to any real ABC Health portfolio.
