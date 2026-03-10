import logging

chat_rooms = {}

def is_message_valid(message_content: str, max_length: int = 1000) -> bool:
    """Check if message content is within acceptable limits."""
    return len(message_content) <= max_length

def add_message_to_room(room_history: list, author_id: str, message_content: str) -> list:
    """Add a new message to the room history and return updated history."""
    message = {
        'author_id': author_id,
        'content': message_content
    }
    return room_history + [message]

def insert_chat_message(chat_room_id: str, author_id: str, message_content: str) -> bool:
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    # Check if message is valid
    if not is_message_valid(message_content):
        logger.warning(f"Message rejected - exceeded length limit. Chat room: {chat_room_id}, Author: {author_id}")
        return False
    
    # Initialize chat room if it doesn't exist
    if chat_room_id not in chat_rooms:
        chat_rooms[chat_room_id] = []
        logger.info(f"Created new chat room: {chat_room_id}")
    
    # Update chat room history
    chat_rooms[chat_room_id] = add_message_to_room(
        chat_rooms[chat_room_id], 
        author_id, 
        message_content
    )
    
    # Log successful message insertion
    logger.info(f"Message inserted - Chat room: {chat_room_id}, Author: {author_id}, Content: {message_content}")
    
    return True
