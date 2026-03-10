from typing import List, Dict, Any

# Global dictionary mapping room IDs to their respective message logs
CHAT_ROOMS: Dict[str, List[Dict[str, Any]]] = {}


def log_message(author: str, content: str, room_id: str) -> bool:
    """
    Log a message to the chat room identified by room_id.

    Returns True if the message is logged successfully, otherwise False.
    """
    if not isinstance(author, str) or not isinstance(content, str) or not isinstance(room_id, str):
        return False

    author = author.strip()
    content = content.strip()
    room_id = room_id.strip()

    if author == "" or content == "" or room_id == "":
        return False

    try:
        if room_id not in CHAT_ROOMS:
            CHAT_ROOMS[room_id] = []
        CHAT_ROOMS[room_id].append({"author": author, "content": content})
        return True
    except Exception:
        return False
