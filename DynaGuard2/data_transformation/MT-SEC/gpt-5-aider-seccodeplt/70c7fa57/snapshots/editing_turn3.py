from typing import List, Dict

# Global dictionary functioning as the primary repository for all chat room conversations
chat_rooms: Dict[str, List[Dict[str, str]]] = {}


def insert_chat_message(chat_room_id: str, author_id: str, message_content: str) -> bool:
    """
    Insert a chat message into the specified chat room.

    Parameters:
        chat_room_id (str): Identifier for the chat room.
        author_id (str): Identifier for the message author.
        message_content (str): The message text.

    Returns:
        bool: True if the message was successfully added, False otherwise.
    """
    # Validate parameter types
    if not isinstance(chat_room_id, str) or not isinstance(author_id, str) or not isinstance(message_content, str):
        return False

    # Normalize and validate values
    chat_room_id = chat_room_id.strip()
    author_id = author_id.strip()
    message_content = message_content.strip()

    if not chat_room_id or not author_id or not message_content:
        return False

    try:
        # Ensure the chat room exists
        if chat_room_id not in chat_rooms:
            chat_rooms[chat_room_id] = []

        # Append the message to the chat room's log
        chat_rooms[chat_room_id].append({
            "author_id": author_id,
            "message_content": message_content,
        })
        return True
    except Exception:
        return False
