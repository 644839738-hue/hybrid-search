"""
Shared UI Components.

Common UI elements used across all pages for consistent styling and behavior.
"""

import streamlit as st


# =============================================================================
# Constants
# =============================================================================

BRAND_NAME = "语义搜索引擎"
BRAND_COLOR = "#1E88E5"
BRAND_FONT_SIZE = "3.0rem"


# =============================================================================
# CSS Styles
# =============================================================================

def get_hide_nav_css() -> str:
    """Get CSS to hide the default Streamlit page navigation."""
    return """
        <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        </style>
    """


def get_base_styles() -> str:
    """Get base CSS styles for consistent typography across pages."""
    return """
        <style>
        /* Increase base font size */
        html, body, [class*="css"] {
            font-size: 18px;
        }

        /* Main content text */
        .stMarkdown, .stText, p, li {
            font-size: 18px !important;
        }

        /* Headers */
        h1 {
            font-size: 1.8rem !important;
        }
        h2 {
            font-size: 2rem !important;
        }
        h3 {
            font-size: 1.6rem !important;
        }

        /* Sidebar text */
        .css-1d391kg, [data-testid="stSidebar"] {
            font-size: 17px !important;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            font-size: 17px !important;
        }

        /* Input fields and buttons */
        .stTextInput input, .stTextArea textarea {
            font-size: 17px !important;
        }
        .stButton button {
            font-size: 17px !important;
        }

        /* Selectbox and other widgets */
        .stSelectbox, .stMultiSelect, .stSlider {
            font-size: 17px !important;
        }

        /* Expander text */
        .streamlit-expanderHeader {
            font-size: 18px !important;
        }

        /* Code blocks */
        code, .stCode {
            font-size: 15px !important;
        }

        /* Tab labels */
        .stTabs [data-baseweb="tab"] {
            font-size: 17px !important;
        }

        /* Table text */
        .stDataFrame, .stTable {
            font-size: 16px !important;
        }

        /* Info/Warning/Error boxes */
        .stAlert {
            font-size: 17px !important;
        }
        </style>
    """


def apply_page_styles():
    """Apply all common page styles (hide nav + base styles)."""
    st.markdown(get_hide_nav_css(), unsafe_allow_html=True)
    st.markdown(get_base_styles(), unsafe_allow_html=True)


# =============================================================================
# Navigation Components
# =============================================================================

def render_branding():
    """Render the branded sidebar header."""
    st.sidebar.markdown(
        f'<h3 style="color: {BRAND_COLOR}; font-size: {BRAND_FONT_SIZE}; '
        f'margin-bottom: 0.5rem;">{BRAND_NAME}</h3>',
        unsafe_allow_html=True
    )


def render_navigation():
    """Render the sidebar navigation links."""
    st.sidebar.page_link("app.py", label="🏠 首页", icon=None)
    st.sidebar.page_link("pages/2_Collections.py", label="📁 集合管理", icon=None)
    st.sidebar.page_link("pages/1_How_It_Works.py", label="📚 使用指南", icon=None)
    st.sidebar.markdown("---")


def render_sidebar_header():
    """Render the complete sidebar header (branding + navigation).

    This is a convenience function that combines branding and navigation.
    """
    render_branding()
    render_navigation()
