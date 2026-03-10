import html


def generate_event_html(event_name: str) -> str:
    """
    Generates a simple HTML heading for an event.

    Args:
        event_name: The name of the event.

    Returns:
        An HTML string representing an <h1> element with the event name.
    """
    return f"<h1>{event_name}</h1>"


def format_event_details_html(description: str, date: str) -> str:
    """
    Formats event description and date into HTML, escaping special characters.

    Args:
        description: The event description.
        date: The event date.

    Returns:
        An HTML string with the event description and date.
    """
    escaped_description = html.escape(description)
    escaped_date = html.escape(date)
    return f"<p>{escaped_description}</p><p><strong>Date:</strong> {escaped_date}</p>"
