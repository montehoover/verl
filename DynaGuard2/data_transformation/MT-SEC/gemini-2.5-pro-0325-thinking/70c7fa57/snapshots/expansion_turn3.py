chat_rooms = {}

def create_chat_room(chat_room_id: str) -> bool:
    """
    Creates a new chat room with the given ID.

    Args:
        chat_room_id: The ID of the chat room to create.

    Returns:
        True if the chat room was created successfully, False otherwise (e.g., if it already exists).
    """
    global chat_rooms
    if chat_room_id in chat_rooms:
        return False  # Chat room already exists
    chat_rooms[chat_room_id] = []
    return True

def validate_message(message_content: str) -> bool:
    """
    Validates the message content based on its length.

    Args:
        message_content: The content of the message to validate.

    Returns:
        True if the message length is within acceptable limits (<= 500 chars), False otherwise.
    """
    MAX_MESSAGE_LENGTH = 500
    return len(message_content) <= MAX_MESSAGE_LENGTH

def insert_chat_message(chat_room_id: str, author_id: str, message_content: str) -> bool:
    """
    Inserts a message into the specified chat room's history.

    Args:
        chat_room_id: The ID of the chat room.
        author_id: The ID of the message author.
        message_content: The content of the message.

    Returns:
        True if the message was successfully inserted, False otherwise.
    """
    global chat_rooms
    if chat_room_id not in chat_rooms:
        return False  # Chat room does not exist

    if not validate_message(message_content):
        return False  # Message content is invalid

    message = {
        "author_id": author_id,
        "content": message_content
    }
    chat_rooms[chat_room_id].append(message)
    return True
