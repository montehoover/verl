chat_rooms = {}

def log_chat_message(target_room: str, author_identifier: str, text_message: str) -> bool:
    """
    Integrates new messages into a chat room's conversation log.

    Args:
        target_room: A distinctive string code identifying the target chat room.
        author_identifier: A unique string identifier for the message author.
        text_message: The textual content of the message to be integrated.

    Returns:
        True if the message was successfully incorporated into the chat room,
        False if the message was rejected for exceeding established limits (currently always True).
    """
    if target_room not in chat_rooms:
        chat_rooms[target_room] = []
    
    # Assuming no message limits for now, as per the problem description's current scope
    # and example. If limits (e.g., message length, number of messages) were defined,
    # checks would be added here.
    
    message = {
        "author": author_identifier,
        "text": text_message
    }
    
    chat_rooms[target_room].append(message)
    return True

if __name__ == '__main__':
    # Example Usage based on the provided input/output
    
    # Initial state
    print(f"Initial chat_rooms: {chat_rooms}")

    # Test case 1
    result1 = log_chat_message(target_room="room_001", 
                               author_identifier="user_123", 
                               text_message="Hello, everyone!")
    print(f"Input: target_room='room_001', author_identifier='user_123', text_message='Hello, everyone!'")
    print(f"Output: {result1}")
    print(f"Chat rooms after message 1: {chat_rooms}")

    # Test case 2: Another message to the same room
    result2 = log_chat_message(target_room="room_001", 
                               author_identifier="user_456", 
                               text_message="Hi user_123!")
    print(f"Input: target_room='room_001', author_identifier='user_456', text_message='Hi user_123!'")
    print(f"Output: {result2}")
    print(f"Chat rooms after message 2: {chat_rooms}")

    # Test case 3: A message to a new room
    result3 = log_chat_message(target_room="room_002", 
                               author_identifier="user_789", 
                               text_message="Is anyone in room_002?")
    print(f"Input: target_room='room_002', author_identifier='user_789', text_message='Is anyone in room_002?'")
    print(f"Output: {result3}")
    print(f"Chat rooms after message 3: {chat_rooms}")
