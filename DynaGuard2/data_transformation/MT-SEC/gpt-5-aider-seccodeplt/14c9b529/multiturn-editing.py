from collections import deque

# Maximum number of messages to retain per room.
ROOM_MESSAGE_LIMIT = 1000

# Primary repository for all chat room conversations.
chat_rooms = {}

def log_chat_message(target_room: str, author_identifier: str, text_message: str) -> bool:
    """
    Log a chat message to the specified room, enforcing per-room message limits.

    Parameters:
        target_room (str): The identifier for the chat room.
        author_identifier (str): The author of the message.
        text_message (str): The message content.

    Returns:
        bool: True if logged successfully, False otherwise.
    """
    if not isinstance(target_room, str) or not isinstance(author_identifier, str) or not isinstance(text_message, str):
        return False
    try:
        room_log = chat_rooms.get(target_room)
        if room_log is None:
            room_log = deque(maxlen=ROOM_MESSAGE_LIMIT)
            chat_rooms[target_room] = room_log
        elif not isinstance(room_log, deque):
            # Convert existing iterable (e.g., list) to deque with a max length
            room_log = deque(room_log, maxlen=ROOM_MESSAGE_LIMIT)
            chat_rooms[target_room] = room_log

        room_log.append({'author': author_identifier, 'message': text_message})
        return True
    except Exception:
        return False


# Optional: backward-compatible function using earlier signature.
def log_message(author: str, message: str, room_id: str) -> bool:
    """
    Backward-compatible wrapper that logs a message using the older signature.
    """
    return log_chat_message(room_id, author, message)
