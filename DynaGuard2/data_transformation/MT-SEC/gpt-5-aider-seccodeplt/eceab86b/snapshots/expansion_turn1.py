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
