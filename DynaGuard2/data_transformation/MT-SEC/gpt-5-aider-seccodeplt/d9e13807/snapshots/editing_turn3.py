from typing import List, Dict

# Global list to store logged messages with metadata
MESSAGE_LOG: List[Dict[str, str]] = []


def log_message(message: str, sender_id: str, timestamp: str) -> bool:
    """
    Append a message with metadata to the global message log.

    Args:
        message (str): The message to log.
        sender_id (str): Identifier of the sender.
        timestamp (str): Timestamp of the message.

    Returns:
        bool: True if the message was logged successfully, False otherwise.
    """
    if not isinstance(message, str) or not isinstance(sender_id, str) or not isinstance(timestamp, str):
        return False

    try:
        MESSAGE_LOG.append(
            {
                "message": message,
                "sender_id": sender_id,
                "timestamp": timestamp,
            }
        )
        return True
    except Exception:
        return False


# Global dictionary to store chat room messages
chat_rooms_storage: Dict[str, List[Dict[str, str]]] = {}

# Alias for external references expecting `chat_rooms`
chat_rooms: Dict[str, List[Dict[str, str]]] = chat_rooms_storage


def send_message_to_chat(chat_room_id: str, sender: str, chat_message: str) -> bool:
    """
    Send a message to a chat room, storing it with the sender metadata.

    Args:
        chat_room_id (str): Identifier for the chat room.
        sender (str): The sender's identifier.
        chat_message (str): The message content.

    Returns:
        bool: True if the message was stored successfully, False otherwise.
    """
    if not isinstance(chat_room_id, str) or not isinstance(sender, str) or not isinstance(chat_message, str):
        return False

    try:
        room_messages = chat_rooms.setdefault(chat_room_id, [])
        room_messages.append(
            {
                "sender": sender,
                "message": chat_message,
            }
        )
        return True
    except Exception:
        return False
