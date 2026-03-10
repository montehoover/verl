import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

chat_rooms = {}

def create_message(member_id: str, content: str) -> dict:
    """Create a message dictionary with member_id and content."""
    return {
        'member_id': member_id,
        'content': content
    }

def ensure_room_exists(room_name: str) -> None:
    """Ensure the chat room exists in the global chat_rooms dictionary."""
    if room_name not in chat_rooms:
        chat_rooms[room_name] = []
        logger.info(f"Created new chat room: {room_name}")

def append_message_to_room(room_name: str, message: dict) -> None:
    """Append a message to the specified chat room."""
    chat_rooms[room_name].append(message)
    logger.info(f"Message added to room '{room_name}' from member '{message['member_id']}': {message['content']}")

def push_text_to_room(room_name: str, member_id: str, content: str) -> bool:
    # Ensure the room exists
    ensure_room_exists(room_name)
    
    # Create the message
    message = create_message(member_id, content)
    
    # Add message to the room
    append_message_to_room(room_name, message)
    
    return True
