from html import escape

def create_event_summary(title: str, date: str, description: str) -> str:
    """
    Create an HTML summary for an event with title, date, and description.
    Each element is placed in its own <div> block and HTML-escaped.

    Args:
        title: The event title.
        date: The event date.
        description: The event description.

    Returns:
        An HTML string containing the formatted event details.
    """
    safe_title = escape("" if title is None else str(title))
    safe_date = escape("" if date is None else str(date))
    safe_description = escape("" if description is None else str(description))

    return (
        f'<div class="event-title">{safe_title}</div>\n'
        f'<div class="event-date">{safe_date}</div>\n'
        f'<div class="event-description">{safe_description}</div>'
    )

def sanitize_notes(notes: str) -> str:
    """
    Sanitize user-provided notes for safe HTML rendering.

    Args:
        notes: The raw notes text provided by the user.

    Returns:
        A string with HTML-sensitive characters escaped.
    """
    return escape("" if notes is None else str(notes))
