chat_rooms = {}  # Dictionary to store messages for multiple rooms

def log_message(room_id: str, author: str, message: str) -> bool:
    """
    Logs a message to a specific chat room.

    Args:
        room_id: The ID of the chat room.
        author: The author of the message.
        message: The content of the message.

    Returns:
        True if the message was logged successfully, False otherwise.
    """
    if not isinstance(room_id, str) or not isinstance(author, str) or not isinstance(message, str):
        return False  # Basic type checking

    try:
        if room_id not in chat_rooms:
            chat_rooms[room_id] = []
        chat_rooms[room_id].append({"author": author, "message": message})
        return True
    except Exception:
        # In a more complex scenario, specific exceptions would be caught.
        return False

if __name__ == '__main__':
    # Example Usage
    print(f"Initial chat rooms: {chat_rooms}")

    # Log messages to 'general' room
    success1 = log_message("general", "Alice", "Hello everyone in general!")
    print(f"Logged 'Hello everyone in general!' from Alice to 'general': {success1}")
    print(f"Chat rooms: {chat_rooms}")

    success2 = log_message("general", "Bob", "Hi Alice!")
    print(f"Logged 'Hi Alice!' from Bob to 'general': {success2}")
    print(f"Chat rooms: {chat_rooms}")

    # Log messages to 'random' room
    success3 = log_message("random", "Charlie", "Anyone here in random?")
    print(f"Logged 'Anyone here in random?' from Charlie to 'random': {success3}")
    print(f"Chat rooms: {chat_rooms}")

    success4 = log_message("general", "Dave", "Welcome to general, Bob!")
    print(f"Logged 'Welcome to general, Bob!' from Dave to 'general': {success4}")
    print(f"Chat rooms: {chat_rooms}")

    # Attempt to log message with invalid type
    success5 = log_message("general", "Eve", 123) # Invalid message type
    print(f"Logged 123 from Eve to 'general': {success5}")
    print(f"Chat rooms: {chat_rooms}")

    # Attempt to log message with invalid room_id type
    success6 = log_message(123, "Frank", "This room ID is invalid") # Invalid room_id type
    print(f"Logged 'This room ID is invalid' from Frank to room 123: {success6}")
    print(f"Chat rooms: {chat_rooms}")
