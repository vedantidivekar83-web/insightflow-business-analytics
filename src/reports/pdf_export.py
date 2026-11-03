"""PDF export module and availability validator for InsightFlow."""

from typing import Any


def is_pdf_export_available() -> bool:
    """Check if a supported PDF rendering engine is installed."""
    try:
        import reportlab  # type: ignore # noqa: F401
        return True
    except ImportError:
        pass

    try:
        import fpdf2  # type: ignore # noqa: F401
        return True
    except ImportError:
        pass

    return False


def get_pdf_limitation_message() -> str:
    """Return a descriptive message detailing the PDF export dependency limitation."""
    return (
        "PDF Report Generation requires the 'reportlab' or 'fpdf2' library, which is not currently "
        "installed in the local environment. In accordance with project architectural guidelines, "
        "no third-party dependencies were installed automatically. Please approve the addition of "
        "'reportlab' to requirements.txt to enable automated executive PDF reporting."
    )
