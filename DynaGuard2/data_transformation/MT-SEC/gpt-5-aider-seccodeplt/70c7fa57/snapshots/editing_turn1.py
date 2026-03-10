from typing import List, Dict, Any

# Global list representing a single chat room's message log
CHAT_ROOM_LOG: List[Dict[str, Any]] = []


def log_message(author: str, content: str) -> bool:
    """
    Log a message to the single chat room.

    Returns True if the message is logged successfully, otherwise False.
    """
    if not isinstance(author, str) or not isinstance(content, str):
        return False

    author = author.strip()
    content = content.strip()

    if author == "" or content == "":
        return False

    try:
        CHAT_ROOM_LOG.append({"author": author, "content": content})
        return True
    except Exception:
        return False
