chat_rooms = {}

def push_text_to_room(room_name: str, member_id: str, content: str) -> bool:
    # Initialize room if it doesn't exist
    if room_name not in chat_rooms:
        chat_rooms[room_name] = []
    
    # Create message dictionary
    message = {
        'member_id': member_id,
        'content': content
    }
    
    # Add message to the room
    chat_rooms[room_name].append(message)
    
    return True
