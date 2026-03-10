chat_room_messages = []

def log_message(author: str, message: str) -> bool:
    """
    Logs a message to the global chat room.

    Args:
        author: The author of the message.
        message: The content of the message.

    Returns:
        True if the message was logged successfully, False otherwise.
    """
    if not isinstance(author, str) or not isinstance(message, str):
        return False  # Basic type checking

    try:
        chat_room_messages.append({"author": author, "message": message})
        return True
    except Exception:
        # In a more complex scenario, specific exceptions would be caught.
        return False

if __name__ == '__main__':
    # Example Usage
    print(f"Initial chat room: {chat_room_messages}")

    success1 = log_message("Alice", "Hello everyone!")
    print(f"Logged 'Hello everyone!' from Alice: {success1}")
    print(f"Chat room: {chat_room_messages}")

    success2 = log_message("Bob", "Hi Alice!")
    print(f"Logged 'Hi Alice!' from Bob: {success2}")
    print(f"Chat room: {chat_room_messages}")

    success3 = log_message("Charlie", 123) # Invalid message type
    print(f"Logged 123 from Charlie: {success3}")
    print(f"Chat room: {chat_room_messages}")
