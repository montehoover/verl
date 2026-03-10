def generate_event_html(event_name: str) -> str:
    """
    Generates a simple HTML heading for an event.

    Args:
        event_name: The name of the event.

    Returns:
        An HTML string representing an <h1> element with the event name.
    """
    return f"<h1>{event_name}</h1>"
