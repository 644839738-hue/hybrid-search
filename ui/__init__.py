"""
UI Components Package.

Shared UI components for consistent styling and behavior across pages.
"""

from ui.shared_components import (
    render_branding,
    render_navigation,
    get_base_styles,
    get_hide_nav_css,
    render_sidebar_header,
    apply_page_styles,
)

from ui.sidebar_components import (
    render_retrieval_settings,
    render_configuration_display,
)

__all__ = [
    # Shared components
    "render_branding",
    "render_navigation",
    "get_base_styles",
    "get_hide_nav_css",
    "render_sidebar_header",
    "apply_page_styles",
    # Sidebar components
    "render_retrieval_settings",
    "render_configuration_display",
]
