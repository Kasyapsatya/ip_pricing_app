"""
IP Pricing Agent — Streamlit Application
==========================================
ABC Health | 9th IAI Capacity Building Seminar in Health and Care Insurance

Dual-interface layout:
  - Sidebar: SSSIA logo, chat history (new/switch/delete, SQLite-persisted), API
    key entry (falls back to a text input only if st.secrets doesn't have one),
    and the deterministic quote-calculator form.
  - Main canvas: quote summary cards (from the deterministic form) + a chat
    interface where "Priya Nair" (an Agno agent wired to the same pricing tools)
    answers follow-up questions, with a real, inspectable tool-call trace shown
    under every answer — not a fabricated summary of what the agent did.

Data is simulated exactly once, offline, via build_data.py -> data/pricing_artifacts.pkl.
This app only ever reads that file; it never re-simulates.
"""
import pickle
from pathlib import Path

import streamlit as st

import chat_store
import pricing_tools as pt

APP_DIR = Path(__file__).parent
ARTIFACTS_PATH = APP_DIR / "data" / "pricing_artifacts.pkl"
LOGO_PATH = APP_DIR / "assets" / "logo.jpeg"

st.set_page_config(page_title="IP Pricing Agent — ABC Health", page_icon="📋", layout="wide")


# ---------------------------------------------------------------------------
# Styling — IAI colour scheme + custom chat bubbles (no default icons)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Open Sans', sans-serif; }

    div.stButton > button:first-child {
        background-color: #005696;
        color: #FFFFFF;
        border: 1px solid #005696;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1.25rem;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #0c4373;
        border-color: #0c4373;
        color: #FFFFFF;
        box-shadow: 0 4px 6px -1px rgba(0, 86, 150, 0.25);
    }

    [data-testid="stSidebar"] {
        background-color: #F0F4F8;
        border-right: 1px solid #CBD5E1;
    }

    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        border-color: #CBD5E1;
        border-radius: 6px;
    }
    div[data-baseweb="input"]:focus-within > div {
        border-color: #005696;
        box-shadow: 0 0 0 1px #005696;
    }

    [data-testid="stMetricValue"] { color: #005696; font-weight: 700; }

    /* --- Custom chat bubbles: Agent vs User boxes, no default avatar icons --- */
    .chat-row { display: flex; margin: 0.6rem 0; }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.agent { justify-content: flex-start; }

    .chat-bubble {
        max-width: 72%;
        padding: 0.7rem 1rem;
        border-radius: 12px;
        line-height: 1.45;
        font-size: 0.95rem;
    }
    .chat-bubble.user {
        background-color: #005696;
        color: #FFFFFF;
        border-bottom-right-radius: 3px;
    }
    .chat-bubble.agent {
        background-color: #FFFFFF;
        color: #1E293B;
        border: 1px solid #CBD5E1;
        border-left: 4px solid #005696;
        border-bottom-left-radius: 3px;
    }

    .chat-avatar {
        width: 34px; height: 34px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.8rem; color: #FFFFFF;
        flex-shrink: 0;
    }
    .chat-avatar.user { background-color: #1E293B; margin-left: 0.5rem; }
    .chat-avatar.agent { background-color: #E5383B; margin-right: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load pricing artifacts once per process (cached across reruns/sessions)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pricing_tools():
    if not ARTIFACTS_PATH.exists():
        st.error(
            f"No pricing data found at `{ARTIFACTS_PATH}`. Run `python build_data.py` "
            f"once before starting the app."
        )
        st.stop()
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    pt.init_from_artifacts(artifacts)
    return pt


tools = load_pricing_tools()


# ---------------------------------------------------------------------------
# API key — st.secrets first, sidebar input only if missing
# ---------------------------------------------------------------------------
def get_api_key():
    secret_key = st.secrets.get("GOOGLE_API_KEY", None) if hasattr(st, "secrets") else None
    if secret_key:
        return secret_key
    return st.session_state.get("manual_api_key")


# ---------------------------------------------------------------------------
# Agno agent — built once per session, after an API key is available
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are Priya Nair, lead pricing actuary at ABC Health, explaining an
Income Protection premium to a colleague or policyholder.

WHAT EACH TOOL MEANS — READ CAREFULLY, THESE ARE NOT INTERCHANGEABLE:

- Tool 1 (age x occupation base rate, via calculate_premium / explain_transition('healthy') /
  explain_occupation): the annual probability that a HEALTHY person in this age/occupation
  group falls sick at all. This is FREQUENCY OF FALLING SICK. It is NOT the probability of
  claiming - most sickness spells never become a paid claim.

- Tool 2 (deferred-period table, via explain_deferred_option): given that a sickness spell has
  occurred, two things - (a) FREQUENCY: what fraction of those spells actually survive the
  deferred period and cross into a paid claim, and (b) SEVERITY: how long the claim runs and
  what it costs, for the deferred period the policyholder actually chose.

- Income scaling (inside calculate_premium only, not a separate tool): Tool 1 and Tool 2's
  numbers are pooled across the whole portfolio. The policyholder's actual income multiplies
  the pooled severity figure up or down to their own income level. This is why
  calculate_premium's output has a field called avg_claim_cost_for_your_income, not
  avg_claim_cost - always use the "for_your_income" figure when discussing THIS policyholder's
  expected claim cost, never the pooled Tool 2 table value directly.

- Tool 3 (episode-based loading, via explain_loading): a personal multiplier on top of
  everything above, based on the policyholder's own prior-episode count. It blends both a
  higher chance of a spell reaching claiming AND a longer claim once it does for people with
  more prior episodes - do not describe it as a pure frequency or pure severity number, it is
  both blended into one factor.

CRITICAL PRECISION RULE - this is a distinct failure mode from inventing numbers, and it
matters just as much:
- NEVER say Tool 1's incidence rate is "the probability of claiming," "the chance you'll need
  a claim," or similar. It is only the probability of FALLING SICK.
- The actual probability of reaching a paid claim is Tool 1's incidence x Tool 2's crossing
  probability, multiplied together - state this explicitly as two separate factors being
  combined, don't collapse them into one figure or badge either one with the other's meaning.
- Citing a real, correctly-sourced number with an incorrect description of what it represents
  is just as much a failure as inventing a number outright. Check every sentence you write
  against what the underlying tool actually measures before saying it.

TOOLS AVAILABLE:
- calculate_premium(age, monthly_income, prior_episodes, occupation, deferred_weeks) -> full premium breakdown
- explain_transition(state_name) -> plain-English explanation of one of the four modelled states
- explain_loading(prior_episode_count) -> plain-English explanation of the experience-rating loading (Tool 3)
- explain_occupation(occupation) -> plain-English explanation of the occupation rating factor (Tool 1)
- explain_deferred_option(deferred_weeks) -> plain-English explanation of the deferred-period effect (Tool 2)
- check_state_exists / check_episode_band_exists / check_occupation_exists / check_deferred_option_exists
  -> guardrails used internally by the above

RULES:
1. Never state a transition rate, loading factor, occupation effect, or deferred-period effect
   unless it came from a tool call.
2. If asked about a factor, category, or option that isn't in the tables (an occupation outside
   desk/manual, a non-standard deferred period, a smoker/non-smoker loading, an episode band
   beyond what's credible), say so plainly and refuse to invent a number - do not guess,
   approximate, or make up a "reasonable-sounding" figure, even if it sounds like the kind of
   thing that should have an answer.
3. Always explain premiums by naming each contributing piece separately and correctly: Tool 1's
   frequency-of-falling-sick, Tool 2's frequency-of-claiming and severity for the chosen
   deferred period, the income scaling applied to that severity, and Tool 3's loading for prior
   sickness history - never collapse these into one opaque number, and never let one factor's
   number carry a different factor's meaning.
4. The person you're talking to may reference a quote already calculated on-screen (age,
   income, occupation, deferred period, prior episodes) - if they ask "why" about a number,
   assume they mean that quote unless they specify different details, and call
   calculate_premium yourself with the same inputs to ground your answer.
"""


@st.cache_resource(show_spinner=False)
def build_agent(api_key):
    from agno.agent import Agent
    from agno.models.google import Gemini

    return Agent(
        model=Gemini(id="gemini-3.1-flash-lite", api_key=api_key),
        tools=[
            tools.calculate_premium, tools.explain_transition, tools.explain_loading,
            tools.explain_occupation, tools.explain_deferred_option,
            tools.check_state_exists, tools.check_episode_band_exists,
            tools.check_occupation_exists, tools.check_deferred_option_exists,
        ],
        system_message=SYSTEM_PROMPT,
        markdown=True,
    )


def extract_tool_trace(response):
    """Pulls tool-call info out of an Agno RunResponse's .messages list. Real
    trace of what the agent actually did — never fabricated."""
    trace = []
    for m in getattr(response, "messages", []) or []:
        if getattr(m, "role", None) == "tool":
            trace.append({
                "tool_name": getattr(m, "tool_name", None),
                "tool_args": getattr(m, "tool_args", None),
                "result": getattr(m, "content", None),
            })
    return trace


# ---------------------------------------------------------------------------
# Session state — active chat id
# ---------------------------------------------------------------------------
if "active_chat_id" not in st.session_state:
    existing = chat_store.list_chats()
    st.session_state.active_chat_id = existing[0]["chat_id"] if existing else chat_store.create_chat()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=110)
    st.markdown("### IP Pricing Agent")
    st.caption("ABC Health · 9th IAI Capacity Building Seminar")

    st.divider()

    if st.button("+ New chat", use_container_width=True):
        st.session_state.active_chat_id = chat_store.create_chat()
        st.rerun()

    st.caption("Chat history")
    for chat in chat_store.list_chats():
        col_a, col_b = st.columns([5, 1])
        is_active = chat["chat_id"] == st.session_state.active_chat_id
        label = ("**➤ " if is_active else "") + chat["title"] + ("**" if is_active else "")
        if col_a.button(label, key=f"chat_{chat['chat_id']}", use_container_width=True):
            st.session_state.active_chat_id = chat["chat_id"]
            st.rerun()
        if col_b.button("🗑", key=f"del_{chat['chat_id']}"):
            chat_store.delete_chat(chat["chat_id"])
            if st.session_state.active_chat_id == chat["chat_id"]:
                remaining = chat_store.list_chats()
                st.session_state.active_chat_id = remaining[0]["chat_id"] if remaining else chat_store.create_chat()
            st.rerun()

    st.divider()

    api_key = get_api_key()
    if not api_key:
        st.warning("No GOOGLE_API_KEY found in secrets.")
        manual_key = st.text_input("Enter your Gemini API key", type="password", key="manual_api_key_input")
        if manual_key:
            st.session_state.manual_api_key = manual_key
            st.rerun()
        api_key = get_api_key()
    else:
        st.success("API key loaded.", icon="✅")

    st.divider()

    # ------------------------ Deterministic quote form ------------------------
    st.markdown("### 📋 Quote Calculator")
    with st.form("quote_form"):
        age = st.number_input("Age", min_value=25, max_value=60, value=45, step=1)
        income = st.number_input("Monthly Income (₹)", min_value=10000, value=80000, step=5000)
        occupation = st.selectbox("Occupation", options=tools.OCCUPATION_CLASSES, index=0)
        deferred_weeks = st.selectbox(
            "Deferred Period (weeks)", options=tools.STANDARD_DEFERRED_OPTIONS,
            index=tools.STANDARD_DEFERRED_OPTIONS.index(13),
        )
        prior_episodes_label = st.selectbox("Prior Sickness Episodes", options=["0", "1", "2+"], index=0)
        submitted = st.form_submit_button("Calculate Quote", use_container_width=True)

    if submitted:
        prior_episodes = 2 if prior_episodes_label == "2+" else int(prior_episodes_label)
        st.session_state.last_quote = tools.calculate_premium(
            age=age, monthly_income=income, prior_episodes=prior_episodes,
            occupation=occupation, deferred_weeks=deferred_weeks,
        )


# ---------------------------------------------------------------------------
# MAIN CANVAS
# ---------------------------------------------------------------------------
st.markdown("## Income Protection — Quote & Explainer")

# --- Quote summary cards ---
quote = st.session_state.get("last_quote")
if quote:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Annual Premium", f"₹{quote['final_annual_premium']:,.0f}")
    c2.metric("Base Premium", f"₹{quote['base_premium']:,.0f}")
    c3.metric("Experience Loading", f"{quote['loading_factor']:.3f}x")
    c4.metric("Income Scale", f"{quote['income_scale']:.3f}x")
    with st.expander("Full breakdown"):
        st.json(quote)
else:
    st.info("Fill in the Quote Calculator in the sidebar and click **Calculate Quote** to see a premium here.")

st.divider()

# --- Chat interface ---
st.markdown("### Ask Priya Nair")
st.caption("e.g. \"Why did my deferred period increase the price?\" or \"What does the loading factor mean?\"")

messages = chat_store.get_messages(st.session_state.active_chat_id)
for msg in messages:
    role_class = "user" if msg["role"] == "user" else "agent"
    avatar_label = "You" if role_class == "user" else "PN"
    avatar_html = f'<div class="chat-avatar {role_class}">{avatar_label}</div>'
    bubble_html = f'<div class="chat-bubble {role_class}">{msg["content"]}</div>'
    if role_class == "user":
        st.markdown(f'<div class="chat-row user">{bubble_html}{avatar_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-row agent">{avatar_html}{bubble_html}</div>', unsafe_allow_html=True)
        if msg.get("tool_trace"):
            with st.expander("🔧 Tool trace"):
                for call in msg["tool_trace"]:
                    st.markdown(f"**`{call['tool_name']}`**")
                    st.code(f"args: {call['tool_args']}\nresult: {call['result']}", language=None)

user_input = st.chat_input("Ask a question about your Income Protection quote...")

if user_input:
    if not api_key:
        st.error("Please provide a Gemini API key in the sidebar before chatting.")
    else:
        is_first_message = len(messages) == 0
        chat_store.add_message(st.session_state.active_chat_id, "user", user_input)
        if is_first_message:
            chat_store.rename_chat_from_first_message(st.session_state.active_chat_id, user_input)

        with st.spinner("Priya is thinking..."):
            agent = build_agent(api_key)
            response = agent.run(user_input)
            trace = extract_tool_trace(response)

        chat_store.add_message(
            st.session_state.active_chat_id, "assistant", response.content, tool_trace=trace,
        )
        st.rerun()
