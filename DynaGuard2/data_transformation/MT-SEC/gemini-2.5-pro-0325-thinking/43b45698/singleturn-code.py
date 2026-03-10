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
        True if the message was successfully added to the chat room, 
        False if the message was rejected due to exceeding defined limits (currently no limits are defined).
    """
    global chat_rooms

    if room_name not in chat_rooms:
        chat_rooms[room_name] = []
    
    # Assuming no limits for now as per problem description.
    # If limits were defined (e.g., max messages per room, max content length),
    # checks would be added here, and the function could return False.

    message = {
        "member_id": member_id,
        "content": content
    }
    
    chat_rooms[room_name].append(message)
    
    return True

if __name__ == '__main__':
    # Example Usage based on the provided input/output
    
    # Initial state
    print(f"Initial chat_rooms: {chat_rooms}")

    # Test case 1
    room_name_1 = "room_001"
    member_id_1 = "user_123"
    content_1 = "Hello, everyone!"
    result_1 = push_text_to_room(room_name_1, member_id_1, content_1)
    print(f"\nInput: room_name='{room_name_1}', member_id='{member_id_1}', content='{content_1}'")
    print(f"Output: {result_1}")
    print(f"chat_rooms after message 1: {chat_rooms}")

    # Test case 2: Adding another message to the same room
    member_id_2 = "user_456"
    content_2 = "Hi user_123!"
    result_2 = push_text_to_room(room_name_1, member_id_2, content_2)
    print(f"\nInput: room_name='{room_name_1}', member_id='{member_id_2}', content='{content_2}'")
    print(f"Output: {result_2}")
    print(f"chat_rooms after message 2: {chat_rooms}")

    # Test case 3: Adding a message to a new room
    room_name_2 = "room_002"
    member_id_3 = "user_789"
    content_3 = "Is anyone in room_002?"
    result_3 = push_text_to_room(room_name_2, member_id_3, content_3)
    print(f"\nInput: room_name='{room_name_2}', member_id='{member_id_3}', content='{content_3}'")
    print(f"Output: {result_3}")
    print(f"chat_rooms after message 3: {chat_rooms}")
