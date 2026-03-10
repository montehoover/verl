"""
Chat message storage and sending utility.

This module provides a small in-memory facility for storing chat messages per
chat room. It exposes the main function `send_message_to_chat` which validates
and appends messages to the in-memory store. The validation and storage logic
is separated into pure helper functions to improve testability and
maintainability.

Globals:
    chat_rooms_storage (dict): The primary in-memory storage for messages.
        Structure:
            {
                "<chat_room_id>": [
                    {
                        "sender": str,
                        "message": str,
                        "timestamp": int  # epoch seconds
                    },
                    ...
                ]
            }
    chat_rooms (dict): Alias to `chat_rooms_storage` provided for external
        compatibility with existing code that expects a `chat_rooms` symbol.

Logging:
    This module emits logs to the module-level logger named after `__name__`.
    By default, a NullHandler is attached so that importing this module does not
    configure global logging. Applications should configure logging handlers at
    the application level to capture these logs.

    Emitted events include:
    - DEBUG when attempting to send a message.
    - WARNING when a message is rejected (length or capacity limits).
    - INFO when a message is successfully appended, including room, sender,
      and message content.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

# Module logger
logger = logging.getLogger(__name__)
# Avoid "No handler found" warnings in libraries
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# Global storage for chat rooms and their messages.
# Structure: { chat_room_id: [ { "sender": str, "message": str, "timestamp": int }, ... ] }
chat_rooms_storage: Dict[str, List[Dict[str, Any]]] = {}

# Alias provided by additional context.
chat_rooms = chat_rooms_storage

# Limits (can be tuned as needed)
MAX_MESSAGE_LENGTH = 4096
MAX_MESSAGES_PER_ROOM = 10000


def is_message_within_length_limit(
    chat_message: str,
    max_length: int = MAX_MESSAGE_LENGTH,
) -> bool:
    """
    Check whether the message length is within the allowed limit.

    Args:
        chat_message: The message text to validate.
        max_length: Maximum allowed message length in characters.

    Returns:
        True if the message length is less than or equal to max_length,
        otherwise False.
    """
    return len(chat_message) <= max_length


def build_message_record(
    sender: str,
    chat_message: str,
    timestamp: int,
) -> Dict[str, Any]:
    """
    Build a message record dictionary.

    Args:
        sender: Unique identifier for the message sender.
        chat_message: The actual text content of the message.
        timestamp: Epoch timestamp (in seconds) for when the message is
            considered created.

    Returns:
        A dictionary containing the message payload with sender, message text,
        and timestamp fields.
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
    Pure storage logic that returns a new list with the message appended.

    This function does not mutate the input list. It returns a tuple indicating
    whether the append was accepted and the resulting list.

    Args:
        existing_messages: Current list of message records for the room.
        new_message: The new message record to append.
        max_messages: Maximum number of messages allowed in the room.

    Returns:
        A tuple (accepted, new_messages_list):
            - accepted (bool): False if appending would exceed room capacity.
            - new_messages_list (list): The original list if rejected, or a
              new list with the appended message if accepted.
    """
    if len(existing_messages) >= max_messages:
        return False, existing_messages

    # Return a new list instance to avoid mutating the input list.
    return True, [*existing_messages, new_message]


def send_message_to_chat(
    chat_room_id: str,
    sender: str,
    chat_message: str,
) -> bool:
    """
    Append a new message to a chat room's history.

    This function is the main entry point for adding messages. It validates the
    message length, constructs a message record with a timestamp, and appends
    it to the in-memory room history while respecting capacity limits.

    Args:
        chat_room_id: Unique identifier for the target chat room.
        sender: Unique identifier for the message sender.
        chat_message: The actual text content of the message.

    Returns:
        True if the message was successfully added to the chat room, False if
        it was rejected due to exceeding limits.

    Raises:
        TypeError: If any of the inputs are not of type str.
    """
    if not isinstance(chat_room_id, str):
        raise TypeError("chat_room_id must be a string")
    if not isinstance(sender, str):
        raise TypeError("sender must be a string")
    if not isinstance(chat_message, str):
        raise TypeError("chat_message must be a string")

    logger.debug(
        "Attempting to send message: room=%s, sender=%s, length=%d",
        chat_room_id,
        sender,
        len(chat_message),
    )

    # Validate message length
    if not is_message_within_length_limit(chat_message):
        logger.warning(
            "Rejected message due to length limit: room=%s, sender=%s, "
            "length=%d, max=%d",
            chat_room_id,
            sender,
            len(chat_message),
            MAX_MESSAGE_LENGTH,
        )
        return False

    # Fetch existing messages (or empty list if room does not exist yet)
    room_messages = chat_rooms_storage.get(chat_room_id, [])

    # Build the message record with a current timestamp
    timestamp = int(time.time())
    message_record = build_message_record(sender, chat_message, timestamp)

    # Attempt to append using pure storage logic
    accepted, new_room_messages = append_message_pure(
        room_messages,
        message_record,
    )
    if not accepted:
        logger.warning(
            "Rejected message due to room capacity: room=%s, sender=%s, "
            "messages=%d, max=%d",
            chat_room_id,
            sender,
            len(room_messages),
            MAX_MESSAGES_PER_ROOM,
        )
        return False

    # Persist the new list back to storage
    chat_rooms_storage[chat_room_id] = new_room_messages

    logger.info(
        "Message appended: room=%s, sender=%s, message=%r, timestamp=%d",
        chat_room_id,
        sender,
        chat_message,
        timestamp,
    )
    return True
