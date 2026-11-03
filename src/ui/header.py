"""Application header component.

Displays top branding, workspace title, and active dataset status.
Enforces the 'Futuristic Purple Glass Analytics Studio' visual aesthetic.
"""

import streamlit as st


def render_header(current_page: str | None = None) -> None:
    """Render the top command bar for InsightFlow with active context."""
    metadata = st.session_state.get("dataset_metadata")
    if metadata and metadata.get("filename"):
        fname = str(metadata["filename"])
        if len(fname) > 22:
            fname = fname[:19] + "..."
        dataset_pill = (
            f'<div style="display: flex; align-items: center; gap: 0.5rem;">'
            f'<span class="if-badge if-badge-cyan" title="{metadata.get("filename")}">'
            f'📁 {fname} &bull; {metadata.get("total_rows", 0):,} rows</span>'
            f'<span class="if-badge if-badge-pink">'
            f'⚡ {metadata.get("total_columns", 0)} dimensions</span></div>'
        )
    else:
        dataset_pill = (
            '<span class="if-badge if-badge-neutral">'
            '📁 No Dataset Loaded &bull; Awaiting Ingestion</span>'
        )

    page_display = f" &bull; {current_page}" if current_page else ""

    header_html = (
        '<div class="if-header-container">'
        '<div style="display: flex; align-items: center; gap: 0.95rem;">'
        '<div style="display: inline-flex; align-items: center; justify-content: center;'
        ' width: 40px; height: 40px; border-radius: 12px;'
        ' background: linear-gradient(135deg, #8B5CF6 0%, #A855F7 50%, #EC4899 100%);'
        ' color: #FFFFFF; font-size: 1.25rem; box-shadow: 0 4px 16px rgba(168, 85, 247, 0.45);'
        ' border: 1px solid rgba(255, 255, 255, 0.2);">⚡</div>'
        '<div><div style="display: flex; align-items: baseline; gap: 0.5rem;">'
        '<h1 class="if-brand-title">InsightFlow</h1>'
        '<span style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase;'
        ' letter-spacing: 0.08em; padding: 0.15rem 0.55rem; border-radius: 6px;'
        ' background: rgba(139, 92, 246, 0.2); color: #C4B5FD;'
        ' border: 1px solid rgba(168, 85, 247, 0.4);">Purple Glass Studio</span>'
        '</div>'
        f'<p class="if-brand-tagline">Autonomous Decision Support &amp; Predictive Analytics Workspace{page_display}</p>'
        '</div></div>'
        f'<div style="display: flex; align-items: center; gap: 0.65rem;">{dataset_pill}</div>'
        '</div>'
    )

    st.markdown(header_html, unsafe_allow_html=True)
