# Global dictionary to store chat messages for multiple rooms
# Key: chat_room_id (str), Value: list of messages (dicts)
chat_rooms = {}

def insert_chat_message(chat_room_id: str, author_id: str, message_content: str) -> bool:
    """
    Inserts a chat message into the specified chat room.

    Args:
        chat_room_id: The identifier of the chat room.
        author_id: The identifier of the message author.
        message_content: The content of the message.

    Returns:
        True if the message was successfully added, False otherwise.
    """
    if not isinstance(chat_room_id, str) or \
       not isinstance(author_id, str) or \
       not isinstance(message_content, str):
        # Basic type checking
        return False
    
    message = {"author_id": author_id, "content": message_content}
    
    # If the room doesn't exist, create it
    if chat_room_id not in chat_rooms:
        chat_rooms[chat_room_id] = []
        
    chat_rooms[chat_room_id].append(message)
    return True

if __name__ == '__main__':
    # Example usage:
    # Insert messages to "general" room
    if insert_chat_message("general", "user_alice", "Hello everyone in general!"):
        print("Message 1 (general) inserted.")
    else:
        print("Failed to insert message 1 (general).")

    if insert_chat_message("general", "user_bob", "Hi Alice, welcome to general!"):
        print("Message 2 (general) inserted.")
    else:
        print("Failed to insert message 2 (general).")

    # Insert messages to "random" room
    if insert_chat_message("random", "user_charlie", "Anyone here in random?"):
        print("Message 3 (random) inserted.")
    else:
        print("Failed to insert message 3 (random).")

    if insert_chat_message("random", "user_dave", "Yes, I'm here in random!"):
        print("Message 4 (random) inserted.")
    else:
        print("Failed to insert message 4 (random).")

    print("\nCurrent chat room messages:")
    for room_id, messages in chat_rooms.items():
        print(f"\n--- Room: {room_id} ---")
        for msg in messages:
            print(f"- {msg['author_id']}: {msg['content']}")
    
    # Example of a failed insertion (e.g. due to incorrect type)
    if insert_chat_message("general", 123, "This should fail"): # 123 is not a string author_id
        print("Message 5 (general) inserted (unexpected).")
    else:
        print("\nFailed to insert message 5 (general) due to incorrect author_id type (as expected).")
    
    if insert_chat_message(None, "user_eve", "This should also fail"): # None is not a string chat_room_id
        print("Message 6 inserted (unexpected).")
    else:
        print("Failed to insert message 6 due to incorrect chat_room_id type (as expected).")
