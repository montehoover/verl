"""
Chat message storage and sending utility.
"""

import time
from typing import Dict, List, Any

# Global storage for chat rooms and their messages.
# Structure: { chat_room_id: [ { "sender": str, "message": str, "timestamp": int }, ... ] }
chat_rooms_storage: Dict[str, List[Dict[str, Any]]] = {}

# Alias provided by additional context.
chat_rooms = chat_rooms_storage

# Limits (can be tuned as needed)
MAX_MESSAGE_LENGTH = 4096
MAX_MESSAGES_PER_ROOM = 10000


def send_message_to_chat(chat_room_id: str, sender: str, chat_message: str) -> bool:
    """
    Appends a new message to a chat room's history.

    Args:
        chat_room_id (str): Unique identifier for the target chat room.
        sender (str): Unique identifier for the message sender.
        chat_message (str): The actual text content of the message.

    Returns:
        bool: True if the message was successfully added to the chat room,
              False if it was rejected due to exceeding limits.
    """
    if not isinstance(chat_room_id, str):
        raise TypeError("chat_room_id must be a string")
    if not isinstance(sender, str):
        raise TypeError("sender must be a string")
    if not isinstance(chat_message, str):
        raise TypeError("chat_message must be a string")

    # Enforce message length limit
    if len(chat_message) > MAX_MESSAGE_LENGTH:
        return False

    # Get or create the room message list
    room_messages = chat_rooms_storage.setdefault(chat_room_id, [])

    # Enforce room capacity limit
    if len(room_messages) >= MAX_MESSAGES_PER_ROOM:
        return False

    # Append the message with a timestamp to facilitate real-time ordering
    room_messages.append(
        {
            "sender": sender,
            "message": chat_message,
            "timestamp": int(time.time()),
        }
    )

    return True
