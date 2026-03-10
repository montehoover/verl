from typing import Dict
from datetime import datetime

chat_rooms = {}

def create_message(member_id: str, content: str) -> Dict[str, any]:
    return {
        'member_id': member_id,
        'content': content,
        'timestamp': datetime.now().isoformat()
    }

def add_message_to_chat_room(room_name: str, message: dict, max_messages: int) -> bool:
    if room_name not in chat_rooms:
        chat_rooms[room_name] = []
    
    if len(chat_rooms[room_name]) >= max_messages:
        return False
    
    chat_rooms[room_name].append(message)
    return True
