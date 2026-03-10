# Global dictionary to store chat messages for multiple rooms
# Key: room_id (str), Value: list of messages (dicts)
chat_rooms = {}

def log_message(room_id: str, author: str, content: str) -> bool:
    """
    Logs a message to the specified chat room.

    Args:
        room_id: The identifier of the chat room.
        author: The name of the message author.
        content: The content of the message.

    Returns:
        True if the message was logged successfully, False otherwise.
    """
    if not isinstance(room_id, str) or not isinstance(author, str) or not isinstance(content, str):
        # Basic type checking
        return False
    
    message = {"author": author, "content": content}
    
    # If the room doesn't exist, create it
    if room_id not in chat_rooms:
        chat_rooms[room_id] = []
        
    chat_rooms[room_id].append(message)
    return True

if __name__ == '__main__':
    # Example usage:
    # Log messages to "general" room
    if log_message("general", "Alice", "Hello everyone in general!"):
        print("Message 1 (general) logged.")
    else:
        print("Failed to log message 1 (general).")

    if log_message("general", "Bob", "Hi Alice, welcome to general!"):
        print("Message 2 (general) logged.")
    else:
        print("Failed to log message 2 (general).")

    # Log messages to "random" room
    if log_message("random", "Charlie", "Anyone here in random?"):
        print("Message 3 (random) logged.")
    else:
        print("Failed to log message 3 (random).")

    if log_message("random", "Dave", "Yes, I'm here in random!"):
        print("Message 4 (random) logged.")
    else:
        print("Failed to log message 4 (random).")

    print("\nCurrent chat room messages:")
    for room_id, messages in chat_rooms.items():
        print(f"\n--- Room: {room_id} ---")
        for msg in messages:
            print(f"- {msg['author']}: {msg['content']}")
    
    # Example of a failed log (e.g. due to incorrect type)
    if log_message("general", 123, "This should fail"):
        print("Message 5 (general) logged (unexpected).")
    else:
        print("\nFailed to log message 5 (general) due to incorrect author type (as expected).")
    
    if log_message(None, "Eve", "This should also fail"):
        print("Message 6 logged (unexpected).")
    else:
        print("Failed to log message 6 due to incorrect room_id type (as expected).")
