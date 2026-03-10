CHAT_ROOMS = {}

def log_message(author: str, message: str, room_id: str) -> bool:
    """
    Append a message to the specified chat room log.

    Parameters:
        author (str): The author of the message.
        message (str): The message content.
        room_id (str): The identifier for the chat room.

    Returns:
        bool: True if logged successfully, False otherwise.
    """
    if not isinstance(author, str) or not isinstance(message, str) or not isinstance(room_id, str):
        return False
    try:
        if room_id not in CHAT_ROOMS:
            CHAT_ROOMS[room_id] = []
        CHAT_ROOMS[room_id].append({'author': author, 'message': message})
        return True
    except Exception:
        return False
