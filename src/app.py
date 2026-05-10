import base64
import html
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from detector import PIIDetector
from coref_resolver import CoreferenceResolver
from entity_merger import EntityChainMerger
from masker import PIIMasker

# page config

st.set_page_config(
    page_title="PII Masking Tool",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# pic


def load_watermark_b64():
    path = Path(__file__).parent / "images/watermark.png"
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


WATERMARK_B64 = load_watermark_b64()

# styling

watermark_css = ""
if WATERMARK_B64:
    watermark_css = f"""
    .watermark {{
        position: fixed;
        bottom: 20px;
        left: 700px;
        max-width: 100px;
        opacity: 0.08;
        pointer-events: none;
        z-index: 0;
    }}
    """

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    }}
    .stApp {{
        background-color: #faf8f3;
        color: #1a1a1a;
    }}
    h1, h2, h3 {{
        font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
        font-weight: 600;
        letter-spacing: -0.01em;
    }}
    h1 {{
        font-size: 2.6rem;
        border-bottom: 2px solid #1a1a1a;
        padding-bottom: 0.4rem;
        margin-bottom: 0.2rem;
    }}
    .subtitle {{
        font-style: italic;
        color: #555;
        margin-bottom: 2rem;
        font-size: 1.05rem;
    }}
    .pane-header {{
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-size: 0.75rem;
        font-family: "SF Mono", "Menlo", "Consolas", monospace;
        color: #777;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #d4cec0;
    }}
    .doc-pane {{
        background: #ffffff;
        border: 1px solid #d4cec0;
        padding: 1.5rem 1.75rem;
        font-family: "SF Mono", "Menlo", "Consolas", monospace;
        font-size: 0.9rem;
        line-height: 1.7;
        white-space: pre-wrap;
        word-wrap: break-word;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
        color: #1a1a1a;
    }}
    .ent {{
        padding: 1px 4px;
        border-radius: 2px;
        font-weight: 500;
        border-bottom: 2px solid;
    }}
    section[data-testid="stSidebar"] {{
        background-color: #f3efe5;
        border-right: 1px solid #d4cec0;
    }}
    section[data-testid="stSidebar"] h2 {{
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
    }}
    .stButton > button {{
        background: #1a1a1a;
        color: #faf8f3;
        border: none;
        border-radius: 0;
        padding: 0.6rem 1.5rem;
        font-family: "SF Mono", "Menlo", monospace;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 500;
    }}
    .stButton > button:hover {{
        background: #8b1a1a;
        color: #faf8f3;
    }}
    .stTextArea textarea {{
        background: #ffffff;
        border: 1px solid #d4cec0;
        border-radius: 0;
        font-family: "SF Mono", "Menlo", "Consolas", monospace;
        font-size: 0.9rem;
        color: #1a1a1a;
    }}
    [data-testid="stMetricValue"] {{
        font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
        font-size: 2rem;
        font-weight: 600;
    }}
    [data-testid="stMetricLabel"] {{
        font-family: "SF Mono", "Menlo", monospace;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.7rem;
        color: #777;
    }}
    .stDataFrame {{
        border: 1px solid #d4cec0;
    }}
    #MainMenu, footer, header {{visibility: hidden;}}
    .legend-chip {{
        display: inline-block;
        padding: 2px 10px;
        margin-right: 8px;
        margin-bottom: 4px;
        font-family: "SF Mono", "Menlo", monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border: 1px solid;
    }}
    .mapping-card {{
        background: #ffffff;
        border: 1px solid #d4cec0;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
        font-family: "SF Mono", "Menlo", "Consolas", monospace;
        font-size: 0.85rem;
    }}
    .mapping-arrow {{
        color: #8b1a1a;
        margin: 0 0.5rem;
        font-weight: 600;
    }}
    .mapping-original {{
        color: #1a1a1a;
        font-weight: 600;
    }}
    .mapping-fake {{
        color: #064e3b;
        font-weight: 600;
    }}
    {watermark_css}
    </style>
    """,
    unsafe_allow_html=True,
)

if WATERMARK_B64:
    st.markdown(
        f'<img class="watermark" src="data:image/png;base64,{WATERMARK_B64}" alt="" />',
        unsafe_allow_html=True,
    )

# colours for lables

LABEL_COLOURS = {
    "PERSON": {"bg": "#fef3c7", "fg": "#78350f", "border": "#b45309"},
    "EMAIL": {"bg": "#dbeafe", "fg": "#1e3a8a", "border": "#2563eb"},
    "PHONE": {"bg": "#fee2e2", "fg": "#7f1d1d", "border": "#dc2626"},
    "EIRCODE": {"bg": "#ede9fe", "fg": "#4c1d95", "border": "#7c3aed"},
    "PPS_NUMBER": {"bg": "#d1fae5", "fg": "#064e3b", "border": "#059669"},
    "IBAN": {"bg": "#ffe4e6", "fg": "#881337", "border": "#be123c"},
    "USERNAME": {"bg": "#e5e7eb", "fg": "#1f2937", "border": "#4b5563"},
    "IP_ADDRESS": {"bg": "#cffafe", "fg": "#155e75", "border": "#0891b2"},
    "CREDIT_CARD": {"bg": "#fce7f3", "fg": "#831843", "border": "#db2777"},
    "API_KEY": {"bg": "#fef9c3", "fg": "#713f12", "border": "#ca8a04"},
}

DEFAULT_COLOUR = {"bg": "#9dbcd4", "fg": "#374151", "border": "#6b7280"}

ALL_LABELS = [
    "PERSON",
    "EMAIL",
    "PHONE",
    "EIRCODE",
    "PPS_NUMBER",
    "IBAN",
    "USERNAME",
    "IP_ADDRESS",
    "CREDIT_CARD",
    "API_KEY",
]

# pipeline loading (cached)


@st.cache_resource(show_spinner="Loading NER and coreference models...")
def load_pipeline():
    return (
        PIIDetector(),
        CoreferenceResolver(),
        EntityChainMerger(),
        PIIMasker(),
    )


# Rendering


def colour_for(label):
    return LABEL_COLOURS.get(label, DEFAULT_COLOUR)


def render_highlighted(text, spans):
    if not spans:
        return f'<div class="doc-pane">{html.escape(text)}</div>'

    spans = sorted(spans, key=lambda s: s["start"])

    out = []
    cursor = 0
    for s in spans:
        start = s["start"]
        end = s["end"]
        # Guard against bad spans silently — just skip them
        if start < cursor or start < 0 or end > len(text) or start >= end:
            continue
        out.append(html.escape(text[cursor:start]))
        c = colour_for(s["label"])
        out.append(
            f'<span class="ent" style="background:{c["bg"]};color:{c["fg"]};'
            f'border-bottom-color:{c["border"]};">'
            f"{html.escape(text[start:end])}"
            f"</span>"
        )
        cursor = end

    out.append(html.escape(text[cursor:]))
    return f'<div class="doc-pane">{"".join(out)}</div>'


def filter_groups_by_label(groups, enabled_labels):
    return [g for g in groups if g.label in enabled_labels]


def copy_button(text_to_copy, label="Copy"):

    safe_text = text_to_copy.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    components.html(
        f"""
        <div style="margin: 0;">
            <button id="copy-btn" onclick="copyText()"
                style="
                    background: #9dbcd4;
                    color: #faf8f3;
                    border: 1px solid #fff;
                    border-radius: 0;
                    padding: 0.6rem 3rem;
                    font-family: 'SF Mono', 'Menlo', monospace;
                    font-size: 1.5rem;
                    text-transform: uppercase;
                    letter-spacing: 0.15em;
                    font-weight: 500;
        cursor: pointer;


        box-shadow: 0 4px 12px rgba(0,0,0,0.12);

        transition: all 0.15s ease;
                ">
                {label}
            </button>
            <span id="copy-status" style="
                margin-left: 12px;
                font-family: 'SF Mono', 'Menlo', monospace;
                font-size: 0.75rem;
                color: #064e3b;
                opacity: 0;
                transition: opacity 0.2s;
            ">Copied</span>
        </div>
        <script>
        const textToCopy = `{safe_text}`;
        function copyText() {{
            const status = document.getElementById('copy-status');
            const showOk = () => {{
                status.style.opacity = 1;
                setTimeout(() => {{ status.style.opacity = 0; }}, 1500);
            }};
            if (navigator.clipboard && window.isSecureContext) {{
                navigator.clipboard.writeText(textToCopy).then(showOk);
            }} else {{
                const ta = document.createElement('textarea');
                ta.value = textToCopy;
                ta.style.position = 'fixed';
                ta.style.left = '-9999px';
                document.body.appendChild(ta);
                ta.select();
                try {{ document.execCommand('copy'); showOk(); }}
                catch (e) {{ status.innerText = 'Copy failed'; status.style.opacity = 1; }}
                document.body.removeChild(ta);
            }}
        }}
        </script>
        """,
        height=80,
    )


# Sample text

SAMPLE_TEXT = """Dear Support Team,

My name is Aoife Murphy and I am writing on behalf of my father, Patrick Murphy, regarding his student account.

Aoife's email address is aoife.murphy@example.ie and her backup email is a.murphy92@gmail.com. You can call her on 087 123 4567, or reach Patrick on +353 1 555 0198.

Their current address is 14 Willow Park Avenue, Rathmines, Dublin 6, D06 F2H3. Patrick previously lived at Apartment 3B, 22 Harbour Road, Galway, H91 X4K7.

Aoife's PPS number is 1234567T. Patrick's PPS number is 7654321W.

For the refund, Aoife provided the following IBAN: IE29 AIBK 9311 5212 3456 78. Patrick's old account used IBAN IE64 BOFI 9058 1234 5678 90.

The online portal usernames are aoife_murphy92, patrick.murphy, and student_A12345.

Yesterday, Aoife spoke with Dr. Niamh O'Brien about the issue. She said Patrick had already emailed Claire Horgan and Andrew Byrne last week, but neither Claire nor Andrew had replied.

Please update the records for Aoife Murphy and Patrick Murphy as soon as possible.

Kind regards,

Aoife Murphy"""

# Sidebar: controls

with st.sidebar:
    st.markdown("## Controls")

    st.markdown("##### PII categories")
    enabled_labels = set()
    for label in ALL_LABELS:
        c = colour_for(label)
        cols = st.columns([1, 4])
        with cols[0]:
            st.markdown(
                f'<div style="width:14px;height:14px;background:{c["bg"]};'
                f'border:2px solid {c["border"]};margin-top:6px;"></div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            if st.checkbox(label, value=True, key=f"toggle_{label}"):
                enabled_labels.add(label)

    st.markdown("---")
    st.caption(
        "Detection uses spaCy + regex. Coreference uses fastcoref. No data leaves your machine."
    )

# Coreference is always on now (toggle removed)
use_coref = True

# MAIN

st.markdown("# PII Masking Tool")
st.markdown(
    '<div class="subtitle">A local, offline tool for redacting personal '
    "information from Irish-context documents.</div>",
    unsafe_allow_html=True,
)

# Input
st.markdown('<div class="pane-header">Input</div>', unsafe_allow_html=True)

tab_paste, tab_sample = st.tabs(["Paste text", "Load sample"])

input_text = ""
with tab_paste:
    input_text = st.text_area(
        "",
        height=200,
        placeholder="Paste the text you want to mask...",
        label_visibility="collapsed",
        key="paste_input",
    )

with tab_sample:
    st.text_area(
        "",
        value=SAMPLE_TEXT,
        height=200,
        disabled=True,
        label_visibility="collapsed",
        key="sample_preview",
    )
    if st.button("Use this sample", key="use_sample"):
        st.session_state["current_text"] = SAMPLE_TEXT

if "current_text" in st.session_state:
    input_text = st.session_state["current_text"]

run = st.button("Run masking", type="primary")

if run and not input_text.strip():
    st.warning("Paste text or load the sample first.")
    st.stop()

if not run:
    st.info("Configure PII categories in the sidebar, provide input, then press Run masking.")
    st.stop()

# Run pipeline

detector, resolver, merger, masker = load_pipeline()

with st.spinner("Detecting entities..."):
    entities = detector.detect(input_text)

with st.spinner("Resolving coreference..."):
    chains = resolver.resolve(input_text)

with st.spinner("Merging and masking..."):
    groups = merger.merge(entities, chains)
    groups = filter_groups_by_label(groups, enabled_labels)
    masked_text, mapping, replacements = masker.mask_with_spans(
        input_text, groups, use_colour=False
    )

# Results

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Entities detected", len(entities))
with m2:
    st.metric("Entity groups", len(groups))
with m3:
    st.metric("Mentions masked", len(replacements))
with m4:
    coref_count = sum(1 for r in replacements if r.source == "coref")
    st.metric("Via coref", coref_count)

st.markdown("")


# Two-pane diff — both sides driven by the masker's authoritative span list
left, right = st.columns(2)

with left:
    st.markdown('<div class="pane-header">Original — detected</div>', unsafe_allow_html=True)
    original_spans = [{"start": r.start, "end": r.end, "label": r.label} for r in replacements]
    st.markdown(
        render_highlighted(input_text, original_spans),
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="pane-header">Masked — replaced</div>', unsafe_allow_html=True)
    masked_spans = [
        {"start": r.masked_start, "end": r.masked_end, "label": r.label} for r in replacements
    ]
    st.markdown(
        render_highlighted(masked_text, masked_spans),
        unsafe_allow_html=True,
    )

st.markdown("")


spacer, btn = st.columns([5, 1])
with btn:
    copy_button(masked_text, label="Copy")

# Mapping table
st.markdown("")
st.markdown('<div class="pane-header">Replacements</div>', unsafe_allow_html=True)

if not replacements:
    st.caption("No entities were masked. Check your category toggles in the sidebar.")
else:
    import pandas as pd

    rows = [
        {
            "Group": r.group_id,
            "Label": r.label,
            "Original": r.original,
            "Replacement": r.fake,
            "Source": r.source,
            "Start": r.start,
            "End": r.end,
        }
        for r in replacements
    ]
    df = pd.DataFrame(rows).sort_values(["Group", "Start"]).reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Group-level summary"):
        summary_rows = []
        for g in groups:
            summary_rows.append(
                {
                    "Group": g.group_id,
                    "Label": g.label,
                    "Anchor": g.anchor,
                    "Fake identity": mapping.get(g.anchor, {}).get("fake", ""),
                    "Mentions": len(g.mentions),
                }
            )
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )
