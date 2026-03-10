chat_rooms = {}

def push_text_to_room(room_name: str, member_id: str, content: str) -> bool:
    """
    Manages the addition of new messages to a chat room's history in a real-time messaging system.
    
    Args:
        room_name: A unique string identifier for the target chat room.
        member_id: A unique string identifier for the user sending the message.
        content: The text content of the message to be added.
    
    Returns:
        bool: Returns True if the message was successfully added to the chat room,
              False if the message was rejected due to exceeding defined limits.
    """
    # Define message content limit (e.g., 1000 characters)
    MAX_MESSAGE_LENGTH = 1000
    
    # Check if message exceeds length limit
    if len(content) > MAX_MESSAGE_LENGTH:
        return False
    
    # Create room if it doesn't exist
    if room_name not in chat_rooms:
        chat_rooms[room_name] = []
    
    # Create message object
    message = {
        'member_id': member_id,
        'content': content
    }
    
    # Add message to the room
    chat_rooms[room_name].append(message)
    
    return True
