import html


def create_event_summary(title: str, date: str, description: str) -> str:
    """
    Formats event details into an HTML string.

    Args:
        title: The title of the event.
        date: The date of the event.
        description: The description of the event.

    Returns:
        An HTML string with the event details in separate div blocks.
    """
    html_output = f"""
<div>{title}</div>
<div>{date}</div>
<div>{description}</div>
"""
    return html_output.strip()


def sanitize_notes(notes: str) -> str:
    """
    Sanitizes a string to make it HTML-safe.

    Args:
        notes: The string to sanitize.

    Returns:
        An HTML-safe version of the input string.
    """
    return html.escape(notes)
