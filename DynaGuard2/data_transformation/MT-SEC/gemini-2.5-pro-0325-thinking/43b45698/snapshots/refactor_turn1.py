chat_rooms = {}

def push_text_to_room(room_name: str, member_id: str, content: str) -> bool:
    """
    Manages the addition of new messages to a chat room's history in a real-time messaging system.
    This function is responsible for appending incoming messages to the appropriate chat room.

    Args:
        room_name: A unique string identifier for the target chat room.
        member_id: A unique string identifier for the user sending the message.
        content: The text content of the message to be added.

    Returns:
        Returns True if the message was successfully added to the chat room,
        False if the message was rejected due to exceeding defined limits (currently always True).
    """
    if room_name not in chat_rooms:
        chat_rooms[room_name] = []
    
    # Assuming no specific limits are defined for message rejection for now.
    # If limits (e.g., max messages per room, content length) were defined,
    # checks would be added here, and the function could return False.

    message = {
        "member_id": member_id,
        "content": content
    }
    chat_rooms[room_name].append(message)
    
    return True
