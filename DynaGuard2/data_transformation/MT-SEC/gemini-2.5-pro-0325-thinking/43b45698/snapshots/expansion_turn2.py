from typing import Dict
from datetime import datetime

def create_message(member_id: str, content: str) -> Dict:
    """
    Creates a message dictionary with member_id, content, and a timestamp.

    Args:
        member_id: The ID of the member sending the message.
        content: The content of the message.

    Returns:
        A dictionary representing the message.
    """
    return {
        "member_id": member_id,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }

chat_rooms: Dict[str, list] = {}

def add_message_to_chat_room(room_name: str, message: Dict, max_messages: int) -> bool:
    """
    Adds a message to a specific chat room's message history.

    Creates the room if it does not exist.
    Enforces a maximum limit on the number of messages held per room.

    Args:
        room_name: The name of the chat room.
        message: The message dictionary to add.
        max_messages: The maximum number of messages to store in the room.

    Returns:
        True if the message was successfully added, False otherwise.
    """
    if room_name not in chat_rooms:
        chat_rooms[room_name] = []
    
    chat_rooms[room_name].append(message)
    
    if len(chat_rooms[room_name]) > max_messages:
        # Remove the oldest messages to maintain the max_messages limit
        num_to_remove = len(chat_rooms[room_name]) - max_messages
        chat_rooms[room_name] = chat_rooms[room_name][num_to_remove:]
        
    return True
