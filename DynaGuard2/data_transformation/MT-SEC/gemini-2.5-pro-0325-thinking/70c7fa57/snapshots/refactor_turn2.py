chat_rooms = {}
MAX_MESSAGES_PER_ROOM = 100  # Example limit: 100 messages per room

def _is_message_limit_exceeded(chat_history: list, max_messages: int) -> bool:
    """Checks if the number of messages in the chat history meets or exceeds the maximum limit."""
    return len(chat_history) >= max_messages

def _append_message_to_history(chat_history: list, author_id: str, message_content: str) -> list:
    """Creates a new message and returns a new list with the message appended to the original history."""
    new_message = {
        "author_id": author_id,
        "content": message_content
    }
    return chat_history + [new_message]

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

    current_room_history = chat_rooms[chat_room_id]

    if _is_message_limit_exceeded(current_room_history, MAX_MESSAGES_PER_ROOM):
        return False  # Message rejected due to room capacity limit

    updated_history = _append_message_to_history(current_room_history, author_id, message_content)
    chat_rooms[chat_room_id] = updated_history
    return True
