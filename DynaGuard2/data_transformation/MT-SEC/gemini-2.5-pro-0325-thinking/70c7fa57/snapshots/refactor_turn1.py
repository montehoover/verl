chat_rooms = {}
MAX_MESSAGES_PER_ROOM = 100  # Example limit: 100 messages per room

def insert_chat_message(chat_room_id: str, author_id: str, message_content: str) -> bool:
    """
    Integrates new messages into a chat room's conversation log.

    Args:
        chat_room_id: A distinctive string code identifying the target chat room.
        author_id: A unique string identifier for the message author.
        message_content: The textual content of the message to be integrated.

    Returns:
        True if the message was successfully incorporated, 
        False if the message was rejected (e.g., for exceeding limits).
    """
    global chat_rooms

    if chat_room_id not in chat_rooms:
        chat_rooms[chat_room_id] = []

    if len(chat_rooms[chat_room_id]) >= MAX_MESSAGES_PER_ROOM:
        return False  # Message rejected due to room capacity limit

    message = {
        "author_id": author_id,
        "content": message_content
    }
    chat_rooms[chat_room_id].append(message)
    return True
