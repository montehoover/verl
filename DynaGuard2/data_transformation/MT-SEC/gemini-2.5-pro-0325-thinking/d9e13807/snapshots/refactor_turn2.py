chat_rooms_storage = {}
MAX_MESSAGES_PER_ROOM = 100

def _is_chat_room_full(chat_room_id: str, storage: dict) -> bool:
    """
    Checks if the chat room has reached its message capacity.
    Pure function: relies only on its inputs.
    """
    if chat_room_id in storage and len(storage[chat_room_id]) >= MAX_MESSAGES_PER_ROOM:
        return True
    return False

def _add_message_to_storage(chat_room_id: str, sender: str, chat_message: str, storage: dict) -> None:
    """
    Adds a message to the chat room's storage.
    Initializes the chat room if it doesn't exist.
    Pure function: modifies a copy of storage or relies on caller to handle storage modification.
    For this refactor, it will modify the passed-in storage dictionary directly.
    """
    if chat_room_id not in storage:
        storage[chat_room_id] = []
    
    message = {
        "sender": sender,
        "message": chat_message
    }
    storage[chat_room_id].append(message)

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
    # Ensure the chat room exists for the check, _is_chat_room_full handles non-existent rooms gracefully
    # by returning False (not full), which is fine.
    # _add_message_to_storage will create it if it doesn't exist.

    if _is_chat_room_full(chat_room_id, chat_rooms_storage):
        return False

    _add_message_to_storage(chat_room_id, sender, chat_message, chat_rooms_storage)
    return True
