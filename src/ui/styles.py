"""Custom CSS styles and UI theming for InsightFlow.

Enforces the 'Futuristic Purple Glass Analytics Studio' visual design system:
- Deep midnight navy canvas (#080817) with translucent glass panels (#17172F, #20203D)
- Refined typography (#F8FAFC headings, #A5A5C5 body, Inter font)
- Futuristic accents: Primary Violet (#8B5CF6), Electric Purple (#A855F7), Cyan (#22D3EE), Pink (#EC4899), Mint (#34D399)
- Glowing active pills, illuminated borders (rgba(167, 139, 250, 0.18)), and ambient glass reflections
"""

import streamlit as st


def apply_custom_styles() -> None:
    """Inject custom CSS rules into the Streamlit document."""
    st.markdown(
        """
        <style>
        /* =========================================================================
           1. Global Typography & Canvas Styling (Futuristic Purple Glass Studio)
           ========================================================================= */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        .main {
            background-color: #080817 !important;
            color: #F8FAFC !important;
        }

        .block-container {
            padding-top: 1.15rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 1340px !important;
        }

        /* Prevent Unwanted Horizontal Scrollbars */
        html, body {
            overflow-x: hidden !important;
            background-color: #080817 !important;
        }

        /* High-Contrast Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #F8FAFC !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
        }

        p, span, div, li {
            color: #A5A5C5;
            line-height: 1.6;
        }

        strong, b {
            color: #F8FAFC !important;
        }

        code {
            color: #22D3EE !important;
            background-color: rgba(32, 32, 61, 0.8) !important;
            border: 1px solid rgba(167, 139, 250, 0.22) !important;
            border-radius: 5px !important;
            padding: 0.15rem 0.45rem !important;
            font-size: 0.85em !important;
        }

        /* =========================================================================
           2. Sidebar Navigation (Purple Glass Studio Shell)
           ========================================================================= */
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"],
        section[data-testid="stSidebar"] {
            background-color: #101024 !important;
            border-right: 1px solid rgba(167, 139, 250, 0.18) !important;
        }

        /* Radio container flex */
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
            display: flex !important;
            flex-direction: column !important;
            gap: 5px !important;
        }

        /* Hide the native radio button circle */
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
            display: none !important;
        }

        /* Navigation items styled as sleek SaaS pills */
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            cursor: pointer !important;
            padding: 0.6rem 0.95rem !important;
            border-radius: 11px !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            border: 1px solid transparent !important;
            margin-bottom: 2px !important;
            background-color: transparent !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background-color: rgba(32, 32, 61, 0.65) !important;
            border-color: rgba(167, 139, 250, 0.25) !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label p {
            color: #A5A5C5 !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            margin: 0 !important;
            transition: color 0.15s ease !important;
        }

        /* Active Navigation Item (:has input:checked) with violet/purple glass glow */
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.28) 0%, rgba(168, 85, 247, 0.16) 100%) !important;
            border: 1px solid rgba(168, 85, 247, 0.55) !important;
            box-shadow: 0 4px 18px -2px rgba(139, 92, 246, 0.35) !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }

        /* =========================================================================
           3. KPI Metrics (Glassmorphic Elevated Metric Cards)
           ========================================================================= */
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(32, 32, 61, 0.75) 0%, rgba(23, 23, 47, 0.85) 100%) !important;
            backdrop-filter: blur(14px) !important;
            -webkit-backdrop-filter: blur(14px) !important;
            border: 1px solid rgba(167, 139, 250, 0.2) !important;
            border-radius: 16px !important;
            padding: 1.15rem 1.35rem !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        [data-testid="stMetric"]:hover {
            border-color: #A855F7 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 28px -2px rgba(168, 85, 247, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        }

        [data-testid="stMetricLabel"] p {
            color: #A5A5C5 !important;
            font-size: 0.74rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            margin: 0 0 0.35rem 0 !important;
        }

        [data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-weight: 700 !important;
            font-size: 1.75rem !important;
            line-height: 1.2 !important;
            letter-spacing: -0.03em !important;
        }

        /* =========================================================================
           4. Tabs (Purple Glass Modern Minimalist)
           ========================================================================= */
        [data-baseweb="tab-list"] {
            gap: 0.45rem !important;
            border-bottom: 1px solid rgba(167, 139, 250, 0.18) !important;
            background-color: transparent !important;
            margin-bottom: 1.35rem !important;
        }

        [data-baseweb="tab"] {
            padding: 0.65rem 1.2rem !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            color: #A5A5C5 !important;
            border-radius: 10px 10px 0 0 !important;
            transition: all 0.15s ease !important;
            border-bottom: 2px solid transparent !important;
            background-color: transparent !important;
        }

        [data-baseweb="tab"]:hover {
            color: #F8FAFC !important;
            background-color: rgba(23, 23, 47, 0.6) !important;
        }

        [data-baseweb="tab"][aria-selected="true"] {
            color: #C4B5FD !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #8B5CF6 !important;
            background: linear-gradient(180deg, rgba(32, 32, 61, 0.4) 0%, rgba(23, 23, 47, 0.8) 100%) !important;
        }

        /* =========================================================================
           5. Containers, Cards, and Expanders
           ========================================================================= */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid rgba(167, 139, 250, 0.18) !important;
            border-radius: 16px !important;
            background-color: rgba(23, 23, 47, 0.75) !important;
            backdrop-filter: blur(14px) !important;
            -webkit-backdrop-filter: blur(14px) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35) !important;
            padding: 1.4rem !important;
        }

        [data-testid="stExpander"] {
            border: 1px solid rgba(167, 139, 250, 0.18) !important;
            border-radius: 14px !important;
            background-color: rgba(23, 23, 47, 0.75) !important;
            backdrop-filter: blur(12px) !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
        }

        [data-testid="stExpander"] summary {
            font-weight: 600 !important;
            color: #F8FAFC !important;
            font-size: 0.92rem !important;
            padding: 0.85rem 1.15rem !important;
        }

        /* =========================================================================
           6. Buttons & Interactive Controls
           ========================================================================= */
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 10px !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.15rem !important;
            border: 1px solid rgba(167, 139, 250, 0.25) !important;
            background: linear-gradient(180deg, rgba(32, 32, 61, 0.9) 0%, rgba(23, 23, 47, 0.9) 100%) !important;
            color: #F8FAFC !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: #A855F7 !important;
            color: #FFFFFF !important;
            background: linear-gradient(180deg, rgba(139, 92, 246, 0.35) 0%, rgba(32, 32, 61, 0.9) 100%) !important;
            box-shadow: 0 6px 20px rgba(168, 85, 247, 0.35) !important;
            transform: translateY(-1px) !important;
        }

        .stButton > button:active,
        .stDownloadButton > button:active {
            transform: scale(0.98) !important;
        }

        .stButton > button:disabled,
        .stDownloadButton > button:disabled {
            border-color: rgba(167, 139, 250, 0.1) !important;
            background-color: rgba(16, 16, 36, 0.6) !important;
            color: #7C7C9E !important;
            box-shadow: none !important;
            opacity: 0.5 !important;
        }

        /* Form Inputs */
        input[type="text"],
        [data-baseweb="select"] > div {
            background-color: #101024 !important;
            border-color: rgba(167, 139, 250, 0.22) !important;
            color: #F8FAFC !important;
            border-radius: 9px !important;
            font-size: 0.88rem !important;
        }

        input[type="text"]:focus,
        [data-baseweb="select"]:focus-within > div {
            border-color: #A855F7 !important;
            box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.25) !important;
        }

        /* Streamlit File Uploader */
        [data-testid="stFileUploader"] {
            border-radius: 16px !important;
        }

        [data-testid="stFileUploader"] section {
            background: linear-gradient(180deg, rgba(16, 16, 36, 0.8) 0%, rgba(23, 23, 47, 0.8) 100%) !important;
            border: 1.5px dashed rgba(167, 139, 250, 0.3) !important;
            border-radius: 16px !important;
            padding: 2.25rem !important;
            transition: all 0.25s ease !important;
        }

        [data-testid="stFileUploader"] section:hover {
            border-color: #A855F7 !important;
            background: rgba(32, 32, 61, 0.7) !important;
            box-shadow: 0 0 28px rgba(168, 85, 247, 0.22) !important;
        }

        /* DataFrames */
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(167, 139, 250, 0.2) !important;
            border-radius: 14px !important;
            overflow: hidden !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
            background-color: #101024 !important;
        }

        /* Streamlit Alerts */
        .stAlert {
            border-radius: 14px !important;
            font-size: 0.88rem !important;
            background-color: rgba(23, 23, 47, 0.85) !important;
            border: 1px solid rgba(167, 139, 250, 0.2) !important;
        }

        /* =========================================================================
           7. Custom Purple Glass Presentation Classes
           ========================================================================= */
        .if-header-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.95rem 1.45rem;
            margin-bottom: 1.6rem;
            background: linear-gradient(90deg, rgba(23, 23, 47, 0.85) 0%, rgba(32, 32, 61, 0.75) 100%);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(167, 139, 250, 0.2);
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }

        .if-brand-title {
            font-size: 1.65rem;
            font-weight: 700;
            color: #F8FAFC !important;
            margin: 0;
            letter-spacing: -0.025em;
            line-height: 1.2;
        }

        .if-brand-tagline {
            font-size: 0.84rem;
            color: #A5A5C5 !important;
            margin-top: 0.2rem;
            margin-bottom: 0;
        }

        .if-hero-banner {
            background: radial-gradient(ellipse at top right, rgba(168, 85, 247, 0.22) 0%, rgba(23, 23, 47, 0.85) 55%, #080817 100%);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(167, 139, 250, 0.22);
            border-radius: 20px;
            padding: 2.25rem 2.6rem;
            margin-bottom: 1.85rem;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }

        .if-hero-title {
            margin: 0.6rem 0 0.5rem 0;
            font-size: 1.95rem;
            color: #F8FAFC !important;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .if-hero-desc {
            margin: 0;
            font-size: 0.96rem;
            color: #A5A5C5 !important;
            line-height: 1.65;
            max-width: 900px;
        }

        .if-card {
            background: rgba(23, 23, 47, 0.75);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(167, 139, 250, 0.18);
            border-radius: 16px;
            padding: 1.4rem 1.55rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .if-card:hover {
            border-color: #A855F7;
            transform: translateY(-3px);
            box-shadow: 0 12px 32px -2px rgba(168, 85, 247, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .if-card-title {
            font-size: 1.02rem;
            font-weight: 600;
            color: #F8FAFC !important;
            margin-bottom: 0.45rem;
            letter-spacing: -0.01em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .if-card-desc {
            font-size: 0.86rem;
            color: #A5A5C5 !important;
            line-height: 1.6;
            margin: 0;
        }

        .if-step-card {
            background: rgba(23, 23, 47, 0.75);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(167, 139, 250, 0.18);
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            transition: all 0.25s ease;
        }

        .if-step-card:hover {
            border-color: #8B5CF6;
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(139, 92, 246, 0.25);
        }

        .if-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.25rem 0.75rem;
            font-size: 0.74rem;
            font-weight: 600;
            border-radius: 9999px;
            background: rgba(139, 92, 246, 0.18);
            color: #C4B5FD !important;
            border: 1px solid rgba(168, 85, 247, 0.38);
            letter-spacing: 0.03em;
        }

        .if-badge-cyan {
            background: rgba(34, 211, 238, 0.14);
            color: #67E8F9 !important;
            border: 1px solid rgba(34, 211, 238, 0.35);
        }

        .if-badge-pink {
            background: rgba(236, 72, 153, 0.14);
            color: #F472B6 !important;
            border: 1px solid rgba(236, 72, 153, 0.35);
        }

        .if-badge-neutral {
            background: rgba(32, 32, 61, 0.7);
            color: #A5A5C5 !important;
            border: 1px solid rgba(167, 139, 250, 0.2);
        }

        .if-badge-success {
            background: rgba(52, 211, 153, 0.14);
            color: #6EE7B7 !important;
            border: 1px solid rgba(52, 211, 153, 0.35);
        }

        .if-badge-warning {
            background: rgba(251, 191, 36, 0.14);
            color: #FCD34D !important;
            border: 1px solid rgba(251, 191, 36, 0.35);
        }

        .if-badge-critical {
            background: rgba(251, 113, 133, 0.14);
            color: #FDA4AF !important;
            border: 1px solid rgba(251, 113, 133, 0.35);
        }

        .if-status-card {
            background: rgba(23, 23, 47, 0.8);
            border: 1px solid rgba(167, 139, 250, 0.18);
            border-radius: 14px;
            padding: 1.05rem 1.15rem;
            font-size: 0.82rem;
            color: #A5A5C5 !important;
            line-height: 1.7;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }

        .if-placeholder-box {
            background: radial-gradient(circle at center, rgba(32, 32, 61, 0.8) 0%, rgba(23, 23, 47, 0.9) 100%);
            border: 1.5px dashed rgba(167, 139, 250, 0.25);
            border-radius: 18px;
            padding: 3.5rem 1.5rem;
            text-align: center;
            margin-top: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .if-page-header {
            margin-bottom: 1.6rem;
            padding-bottom: 0.95rem;
            border-bottom: 1px solid rgba(167, 139, 250, 0.15);
        }

        .if-page-title {
            margin: 0 0 0.25rem 0;
            font-size: 1.65rem;
            color: #F8FAFC;
            font-weight: 700;
            letter-spacing: -0.025em;
        }

        .if-page-subtitle {
            margin: 0;
            font-size: 0.9rem;
            color: #A5A5C5;
            line-height: 1.5;
        }

        .if-callout {
            background: rgba(23, 23, 47, 0.85);
            border: 1px solid rgba(167, 139, 250, 0.22);
            border-left: 3px solid #8B5CF6;
            border-radius: 12px;
            padding: 0.9rem 1.15rem;
            margin: 0.75rem 0 1rem 0;
            font-size: 0.88rem;
            color: #F8FAFC !important;
            line-height: 1.6;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }

        .if-callout code {
            color: #22D3EE !important;
        }

        .if-divider {
            height: 1px;
            background: rgba(167, 139, 250, 0.15);
            margin: 1.25rem 0;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
