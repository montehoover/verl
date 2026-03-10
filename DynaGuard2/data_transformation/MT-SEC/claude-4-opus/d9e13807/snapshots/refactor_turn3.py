import logging

# Configure logging
logger = logging.getLogger(__name__)

# Global storage for chat rooms
chat_rooms_storage = {}


def _validate_message_limit(chat_room_messages: list, limit: int = 1000) -> bool:
    """Check if the chat room has reached its message limit.
    
    Args:
        chat_room_messages (list): List of messages in the chat room.
        limit (int, optional): Maximum number of messages allowed. Defaults to 1000.
    
    Returns:
        bool: True if under limit, False if limit reached.
    """
    return len(chat_room_messages) < limit


def _create_message(sender: str, chat_message: str) -> dict:
    """Create a message object with sender and content.
    
    Args:
        sender (str): Unique identifier for the message sender.
        chat_message (str): The actual text content of the message.
    
    Returns:
        dict: Message object containing sender, message, and timestamp.
    """
    return {
        'sender': sender,
        'message': chat_message,
        'timestamp': None  # Could add timestamp if needed
    }


def _get_or_create_chat_room(chat_room_id: str) -> list:
    """Get existing chat room or create a new one if it doesn't exist.
    
    Args:
        chat_room_id (str): Unique identifier for the target chat room.
    
    Returns:
        list: List of messages for the specified chat room.
    """
    if chat_room_id not in chat_rooms_storage:
        chat_rooms_storage[chat_room_id] = []
        logger.info(f"Created new chat room: {chat_room_id}")
    return chat_rooms_storage[chat_room_id]


def _add_message_to_room(chat_room_messages: list, message: dict) -> None:
    """Add a message to the chat room's message list.
    
    Args:
        chat_room_messages (list): List of messages in the chat room.
        message (dict): Message object to add to the room.
    
    Returns:
        None
    """
    chat_room_messages.append(message)


def send_message_to_chat(chat_room_id: str, sender: str, chat_message: str) -> bool:
    """Append a new message to a specific chat room's history.
    
    This function facilitates real-time communication by adding messages
    to the specified chat room. Messages are stored in memory and subject
    to a configurable limit per room.
    
    Args:
        chat_room_id (str): Unique identifier for the target chat room.
        sender (str): Unique identifier for the message sender.
        chat_message (str): The actual text content of the message.
    
    Returns:
        bool: True if the message was successfully added to the chat room,
              False if it was rejected due to exceeding limits.
    """
    # Get or create chat room
    chat_room_messages = _get_or_create_chat_room(chat_room_id)
    
    # Validate message limit
    if not _validate_message_limit(chat_room_messages):
        logger.warning(
            f"Message rejected - limit exceeded for chat_room_id: {chat_room_id}, "
            f"sender: {sender}"
        )
        return False
    
    # Create and add message
    message = _create_message(sender, chat_message)
    _add_message_to_room(chat_room_messages, message)
    
    # Log successful message addition
    logger.info(
        f"Message added - chat_room_id: {chat_room_id}, "
        f"sender: {sender}, message: {chat_message}"
    )
    
    return True
