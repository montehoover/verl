"""
Chat message storage and sending utility.
"""

import time
from typing import Dict, List, Any, Tuple

# Global storage for chat rooms and their messages.
# Structure: { chat_room_id: [ { "sender": str, "message": str, "timestamp": int }, ... ] }
chat_rooms_storage: Dict[str, List[Dict[str, Any]]] = {}

# Alias provided by additional context.
chat_rooms = chat_rooms_storage

# Limits (can be tuned as needed)
MAX_MESSAGE_LENGTH = 4096
MAX_MESSAGES_PER_ROOM = 10000


def is_message_within_length_limit(chat_message: str, max_length: int = MAX_MESSAGE_LENGTH) -> bool:
    """
    Pure validation: checks if the message length is within the allowed limit.
    """
    return len(chat_message) <= max_length


def build_message_record(sender: str, chat_message: str, timestamp: int) -> Dict[str, Any]:
    """
    Pure construction: builds a message record dict from the provided components.
    """
    return {
        "sender": sender,
        "message": chat_message,
        "timestamp": int(timestamp),
    }


def append_message_pure(
    existing_messages: List[Dict[str, Any]],
    new_message: Dict[str, Any],
    max_messages: int = MAX_MESSAGES_PER_ROOM,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Pure storage logic: returns a tuple (accepted, new_messages_list) without mutating inputs.
    - accepted is False if adding the message would exceed the room capacity.
    """
    if len(existing_messages) >= max_messages:
        return False, existing_messages
    # Return a new list instance to avoid mutating the input list.
    return True, [*existing_messages, new_message]


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

    # Validate message length
    if not is_message_within_length_limit(chat_message):
        return False

    # Fetch existing messages (or empty list if room does not exist yet)
    room_messages = chat_rooms_storage.get(chat_room_id, [])

    # Build the message record with a current timestamp
    message_record = build_message_record(sender, chat_message, int(time.time()))

    # Attempt to append using pure storage logic
    accepted, new_room_messages = append_message_pure(room_messages, message_record)
    if not accepted:
        return False

    # Persist the new list back to storage
    chat_rooms_storage[chat_room_id] = new_room_messages
    return True
