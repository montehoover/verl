chat_rooms_storage = {}
MAX_MESSAGES_PER_ROOM = 100

def send_message_to_chat(chat_room_id: str, sender: str, chat_message: str) -> bool:
    """
    Appends new messages to a specific chat room's history.

    Args:
        chat_room_id: Unique identifier for the target chat room.
        sender: Unique identifier for the message sender.
        chat_message: The actual text content of the message.

    Returns:
        True if the message was successfully added to the chat room,
        False if it was rejected due to exceeding limits.
    """
    if chat_room_id not in chat_rooms_storage:
        chat_rooms_storage[chat_room_id] = []

    if len(chat_rooms_storage[chat_room_id]) >= MAX_MESSAGES_PER_ROOM:
        return False

    message = {
        "sender": sender,
        "message": chat_message
    }
    chat_rooms_storage[chat_room_id].append(message)
    return True
