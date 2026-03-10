chat_rooms = {}
MAX_MESSAGES_PER_ROOM = 100  # Established limit for messages in a room


def _is_room_at_capacity(room_name: str, rooms_data: dict, max_messages: int) -> bool:
    """Checks if the specified chat room has reached its message capacity."""
    return room_name in rooms_data and len(rooms_data[room_name]) >= max_messages


def _append_message_to_room(room_name: str, author: str, text: str, rooms_data: dict):
    """Appends a new message to the specified chat room's log."""
    message = {
        "author": author,
        "text": text
    }
    rooms_data[room_name].append(message)


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

    if _is_room_at_capacity(target_room, chat_rooms, MAX_MESSAGES_PER_ROOM):
        return False  # Message rejected due to exceeding room message limit

    _append_message_to_room(target_room, author_identifier, text_message, chat_rooms)
    return True
