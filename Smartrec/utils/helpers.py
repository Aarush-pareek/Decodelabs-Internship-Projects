"""Small presentation helpers used by the Streamlit UI."""
import html


def esc(value: object) -> str:
    """HTML-escape any value (dataset text is never trusted in HTML)."""
    return html.escape(str(value), quote=True)


def format_percent(score: float) -> str:
    """0.874 -> '87%'."""
    return f"{round(score * 100)}%"


def chips_html(items: list[str], css_class: str = "chip") -> str:
    """Render a list of labels as HTML chips."""
    return "".join(f'<span class="{css_class}">{esc(item)}</span>' for item in items)


def truncate(text: str, limit: int = 140) -> str:
    """Shorten long text without cutting in the middle of a word."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "..."
