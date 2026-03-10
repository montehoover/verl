from typing import Dict, List, Any

# Global repository for all chat room conversations
chat_rooms: Dict[str, List[Dict[str, Any]]] = {}

# Established limits
MAX_MESSAGE_LENGTH = 4096          # Maximum allowed characters in a single message
MAX_ROOM_MESSAGES = 10000          # Maximum number of messages stored per room


def insert_chat_message(chat_room_id: str, author_id: str, message_content: str) -> bool:
    """
    Integrate a new message into the specified chat room's conversation log.

    Args:
        chat_room_id (str): A distinctive string code identifying the target chat room.
        author_id (str): A unique string identifier for the message author.
        message_content (str): The textual content of the message to be integrated.

    Returns:
        bool: True if the message was successfully incorporated; False if the message
              was rejected for exceeding established limits.
    """
    # Enforce message size limit
    if not isinstance(message_content, str) or len(message_content) > MAX_MESSAGE_LENGTH:
        return False

    # Initialize room if it doesn't exist
    room_log = chat_rooms.setdefault(chat_room_id, [])

    # Enforce per-room message count limit
    if len(room_log) >= MAX_ROOM_MESSAGES:
        return False

    # Append the message to the room's log
    room_log.append({
        "author_id": author_id,
        "message_content": message_content,
    })

    return True
