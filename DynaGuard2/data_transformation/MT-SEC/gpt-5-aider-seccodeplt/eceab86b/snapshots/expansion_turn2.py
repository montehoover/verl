from html import escape

def generate_event_html(event_name: str) -> str:
    """
    Generate a simple HTML heading for the event name.

    Args:
        event_name: The name of the event.

    Returns:
        A string containing the event name wrapped in an <h1> element.
    """
    return f"<h1>{escape(event_name)}</h1>"

def format_event_details_html(description: str, date: str) -> str:
    """
    Generate HTML for an event's description and date, with HTML-escaping.

    Args:
        description: The event description text.
        date: The event date text.

    Returns:
        A string with the description wrapped in <p> and the date in
        <p><strong>Date:</strong> ...</p>, with content safely escaped.
    """
    safe_description = escape(description)
    safe_date = escape(date)
    return f"<p>{safe_description}</p><p><strong>Date:</strong> {safe_date}</p>"
