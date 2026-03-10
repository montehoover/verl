import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global chat rooms repository
chat_rooms = {}


def validate_message(text_message: str) -> bool:
    """Validate if a message meets the requirements."""
    return len(text_message) <= 1000


def create_message_entry(author_identifier: str, text_message: str) -> dict:
    """Create a message entry dictionary."""
    return {
        'author': author_identifier,
        'message': text_message
    }


def add_message_to_room(target_room: str, message_entry: dict) -> None:
    """Add a message entry to the specified chat room."""
    if target_room not in chat_rooms:
        chat_rooms[target_room] = []
    
    chat_rooms[target_room].append(message_entry)


def log_chat_message(target_room: str, author_identifier: str, text_message: str) -> bool:
    """
    Integrates new messages into a chat room's conversation log.
    
    Args:
        target_room: A distinctive string code identifying the target chat room.
        author_identifier: A unique string identifier for the message author.
        text_message: The textual content of the message to be integrated.
    
    Returns:
        bool: True if the message was successfully incorporated into the chat room,
              False if the message was rejected for exceeding established limits.
    """
    # Log the attempt
    logger.info(f"Attempting to log message from {author_identifier} to room {target_room}")
    
    # Validate message
    if not validate_message(text_message):
        logger.warning(f"Message from {author_identifier} rejected: exceeded length limit")
        return False
    
    # Create message entry
    message_entry = create_message_entry(author_identifier, text_message)
    
    # Add message to room
    add_message_to_room(target_room, message_entry)
    
    # Log successful integration
    logger.info(f"Message from {author_identifier} successfully added to room {target_room}")
    
    return True
