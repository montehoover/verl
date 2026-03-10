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
