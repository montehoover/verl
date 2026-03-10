from typing import Dict, List

chat_rooms_storage = {}

def create_chat_room(chat_room_id: str) -> bool:
    if chat_room_id in chat_rooms_storage:
        return False
    chat_rooms_storage[chat_room_id] = []
    return True

def add_message_to_room(chat_room_id: str, sender: str, message: str) -> bool:
    if chat_room_id not in chat_rooms_storage:
        return False
    chat_rooms_storage[chat_room_id].append({
        'sender': sender,
        'message': message
    })
    return True

def send_message_to_chat(chat_room_id: str, sender: str, chat_message: str) -> bool:
    if chat_room_id not in chat_rooms_storage:
        return False
    if len(chat_rooms_storage[chat_room_id]) >= 100:  # Assuming a limit of 100 messages per room
        return False
    chat_rooms_storage[chat_room_id].append({
        "sender": sender,
        "message": chat_message
    })
    return True
