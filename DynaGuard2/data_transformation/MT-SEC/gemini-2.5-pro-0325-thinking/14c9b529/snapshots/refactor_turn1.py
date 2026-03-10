chat_rooms = {}
MAX_MESSAGES_PER_ROOM = 100  # Established limit for messages in a room

def log_chat_message(target_room: str, author_identifier: str, text_message: str) -> bool:
    """
    Integrates new messages into a chat room's conversation log.

    Args:
        target_room: A distinctive string code identifying the target chat room.
        author_identifier: A unique string identifier for the message author.
        text_message: The textual content of the message to be integrated.

    Returns:
        True if the message was successfully incorporated into the chat room,
        False if the message was rejected for exceeding established limits.
    """
    global chat_rooms

    if target_room not in chat_rooms:
        chat_rooms[target_room] = []

    if len(chat_rooms[target_room]) >= MAX_MESSAGES_PER_ROOM:
        return False  # Message rejected due to exceeding room message limit

    message = {
        "author": author_identifier,
        "text": text_message
    }
    chat_rooms[target_room].append(message)
    return True
