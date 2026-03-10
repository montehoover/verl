import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

chat_rooms_storage = {}
MAX_MESSAGES_PER_ROOM = 100

def _is_chat_room_full(chat_room_id: str, storage: dict) -> bool:
    """
    Checks if the specified chat room has reached its message capacity.

    This function is pure as it only relies on its inputs and does not
    cause side effects.

    Args:
        chat_room_id: The unique identifier for the chat room.
        storage: The dictionary holding all chat room messages.

    Returns:
        True if the chat room is full, False otherwise.
    """
    if chat_room_id in storage and len(storage[chat_room_id]) >= MAX_MESSAGES_PER_ROOM:
        return True
    return False

def _add_message_to_storage(chat_room_id: str, sender: str, chat_message: str, storage: dict) -> None:
    """
    Adds a message to the specified chat room's storage.

    Initializes the chat room within the storage if it doesn't already exist.
    This function modifies the `storage` dictionary in place by appending
    the new message, which is a side effect.

    Args:
        chat_room_id: The unique identifier for the chat room.
        sender: The unique identifier for the message sender.
        chat_message: The text content of the message.
        storage: The dictionary holding all chat room messages. This dictionary
                 will be modified.
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
        logger.warning(
            f"Message from sender '{sender}' to chat room '{chat_room_id}' "
            f"rejected: Room is full."
        )
        return False

    _add_message_to_storage(chat_room_id, sender, chat_message, chat_rooms_storage)
    
    # For logging, show only a preview of long messages to avoid overly verbose logs
    message_preview = (chat_message[:75] + '...') if len(chat_message) > 75 else chat_message
    logger.info(
        f"Message from sender '{sender}' added to chat room '{chat_room_id}'. "
        f"Preview: '{message_preview}'"
    )
    return True
