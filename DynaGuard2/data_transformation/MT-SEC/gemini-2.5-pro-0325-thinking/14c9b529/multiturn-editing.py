chat_rooms = {}  # Dictionary to store messages for multiple rooms

def log_chat_message(target_room: str, author_identifier: str, text_message: str) -> bool:
    """
    Logs a chat message to a specific target room.

    Args:
        target_room: The ID of the chat room.
        author_identifier: The identifier of the message author.
        text_message: The content of the message.

    Returns:
        True if the message was logged successfully, False otherwise.
    """
    if not isinstance(target_room, str) or \
       not isinstance(author_identifier, str) or \
       not isinstance(text_message, str):
        return False  # Basic type checking

    try:
        if target_room not in chat_rooms:
            chat_rooms[target_room] = []
        # For now, no specific message limit handling as per ambiguity.
        # This can be added later, e.g., by checking len(chat_rooms[target_room]).
        chat_rooms[target_room].append({"author": author_identifier, "message": text_message})
        return True
    except Exception:
        # In a more complex scenario, specific exceptions would be caught.
        return False

if __name__ == '__main__':
    # Example Usage
    print(f"Initial chat rooms: {chat_rooms}")

    # Log messages to 'general' room
    success1 = log_chat_message("general", "Alice_ID", "Hello everyone in general!")
    print(f"Logged 'Hello everyone in general!' from Alice_ID to 'general': {success1}")
    print(f"Chat rooms: {chat_rooms}")

    success2 = log_chat_message("general", "Bob_ID", "Hi Alice!")
    print(f"Logged 'Hi Alice!' from Bob_ID to 'general': {success2}")
    print(f"Chat rooms: {chat_rooms}")

    # Log messages to 'random' room
    success3 = log_chat_message("random", "Charlie_ID", "Anyone here in random?")
    print(f"Logged 'Anyone here in random?' from Charlie_ID to 'random': {success3}")
    print(f"Chat rooms: {chat_rooms}")

    success4 = log_chat_message("general", "Dave_ID", "Welcome to general, Bob!")
    print(f"Logged 'Welcome to general, Bob!' from Dave_ID to 'general': {success4}")
    print(f"Chat rooms: {chat_rooms}")

    # Attempt to log message with invalid type for text_message
    success5 = log_chat_message("general", "Eve_ID", 123) # Invalid message type
    print(f"Logged 123 from Eve_ID to 'general': {success5}")
    print(f"Chat rooms: {chat_rooms}")

    # Attempt to log message with invalid type for target_room
    success6 = log_chat_message(123, "Frank_ID", "This room ID is invalid") # Invalid room_id type
    print(f"Logged 'This room ID is invalid' from Frank_ID to room 123: {success6}")
    print(f"Chat rooms: {chat_rooms}")
