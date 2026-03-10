CHAT_LOG = []

def log_message(author: str, message: str) -> bool:
    """
    Append a message to the global chat log.

    Parameters:
        author (str): The author of the message.
        message (str): The message content.

    Returns:
        bool: True if logged successfully, False otherwise.
    """
    if not isinstance(author, str) or not isinstance(message, str):
        return False
    try:
        CHAT_LOG.append({'author': author, 'message': message})
        return True
    except Exception:
        return False
