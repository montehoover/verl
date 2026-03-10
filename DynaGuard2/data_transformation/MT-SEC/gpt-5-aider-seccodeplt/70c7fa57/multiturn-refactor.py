from typing import Dict, List, Any
import logging

# Global repository for all chat room conversations
chat_rooms: Dict[str, List[Dict[str, Any]]] = {}

# Established limits
MAX_MESSAGE_LENGTH = 4096          # Maximum allowed characters in a single message
MAX_ROOM_MESSAGES = 10000          # Maximum number of messages stored per room

# Module-level logger and initialization flag
_LOGGER = logging.getLogger("chat.messages")
_LOGGING_INITIALIZED = False


def _init_logging_if_needed() -> None:
    """
    Initialize logging with a human-readable format if it hasn't been initialized yet.
    """
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    if not _LOGGER.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        _LOGGER.addHandler(handler)
        _LOGGER.setLevel(logging.INFO)
        # Prevent double logging if root logger is configured elsewhere
        _LOGGER.propagate = False

    _LOGGING_INITIALIZED = True


def within_message_limits(
    message_content: str,
    room_message_count: int,
    max_message_length: int = MAX_MESSAGE_LENGTH,
    max_room_messages: int = MAX_ROOM_MESSAGES,
) -> bool:
    """
    Pure function to validate whether a message can be accepted based on established limits.

    Args:
        message_content (str): The textual content of the message.
        room_message_count (int): Current number of messages in the room.
        max_message_length (int): Maximum allowed characters in a single message.
        max_room_messages (int): Maximum number of messages stored per room.

    Returns:
        bool: True if the message is within limits; False otherwise.
    """
    if not isinstance(message_content, str):
        return False
    if len(message_content) > max_message_length:
        return False
    if room_message_count >= max_room_messages:
        return False
    return True


def next_room_history(
    room_log: List[Dict[str, Any]],
    author_id: str,
    message_content: str,
) -> List[Dict[str, Any]]:
    """
    Pure function that returns a new room history with the message appended.

    Args:
        room_log (List[Dict[str, Any]]): The current room message list.
        author_id (str): Author of the message.
        message_content (str): The message content.

    Returns:
        List[Dict[str, Any]]: A new list representing the updated room history.
    """
    new_entry = {
        "author_id": author_id,
        "message_content": message_content,
    }
    return room_log + [new_entry]


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
    # Initialize logging (idempotent)
    _init_logging_if_needed()

    # Read current room log without mutating global state
    room_log = chat_rooms.get(chat_room_id, [])

    # Enforce all message limits
    if not within_message_limits(message_content, len(room_log)):
        _LOGGER.warning(
            "Rejected message due to limits | room=%s author=%s content_length=%d current_messages=%d",
            chat_room_id,
            author_id,
            len(message_content) if isinstance(message_content, str) else -1,
            len(room_log),
        )
        return False

    # Compute the next room history in a pure way, then commit to global state
    chat_rooms[chat_room_id] = next_room_history(room_log, author_id, message_content)

    _LOGGER.info(
        "Inserted message | room=%s author=%s content=%s",
        chat_room_id,
        author_id,
        message_content,
    )

    return True
