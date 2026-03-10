# Global list to store chat messages
chat_room_messages = []

def log_message(author: str, content: str) -> bool:
    """
    Logs a message to the global chat room.

    Args:
        author: The name of the message author.
        content: The content of the message.

    Returns:
        True if the message was logged successfully, False otherwise.
    """
    if not isinstance(author, str) or not isinstance(content, str):
        # Basic type checking, could be expanded
        return False
    
    message = {"author": author, "content": content}
    chat_room_messages.append(message)
    return True

if __name__ == '__main__':
    # Example usage:
    if log_message("Alice", "Hello everyone!"):
        print("Message 1 logged.")
    else:
        print("Failed to log message 1.")

    if log_message("Bob", "Hi Alice!"):
        print("Message 2 logged.")
    else:
        print("Failed to log message 2.")

    print("\nCurrent chat room messages:")
    for msg in chat_room_messages:
        print(f"- {msg['author']}: {msg['content']}")
    
    # Example of a failed log (e.g. due to incorrect type, though not strictly enforced yet beyond basic check)
    if log_message(123, "This should fail"):
        print("Message 3 logged (unexpected).")
    else:
        print("Failed to log message 3 (as expected).")
