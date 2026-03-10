def generate_event_update(title: str, message: str, date: str) -> str:
    """
    Create a formatted event update string combining title, message, and date.
    """
    title_s = str(title).strip()
    message_s = str(message).strip()
    date_s = str(date).strip()
    return f"{title_s} [{date_s}]: {message_s}"
