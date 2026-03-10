chat_rooms_storage = {}

def send_message_to_chat(chat_room_id: str, sender: str, chat_message: str) -> bool:
    """
    Appends a chat message to the specified chat room in chat_rooms_storage.

    Args:
        chat_room_id: The ID of the chat room.
        sender: The ID of the message sender.
        chat_message: The content of the chat message.

    Returns:
        True if the message was logged successfully, False otherwise.
    """
    try:
        if chat_room_id not in chat_rooms_storage:
            chat_rooms_storage[chat_room_id] = []
        
        message_entry = {
            "sender": sender,
            "message": chat_message
        }
        chat_rooms_storage[chat_room_id].append(message_entry)
        return True
    except Exception:
        # In a real-world scenario, log specific exceptions
        return False

if __name__ == '__main__':
    # Example Usage for send_message_to_chat
    if send_message_to_chat("general", "user1", "Hello everyone!"):
        print("Message 1 to 'general' sent successfully.")
    else:
        print("Failed to send message 1 to 'general'.")

    if send_message_to_chat("general", "user2", "Hi user1!"):
        print("Message 2 to 'general' sent successfully.")
    else:
        print("Failed to send message 2 to 'general'.")

    if send_message_to_chat("random", "user3", "Anyone here?"):
        print("Message 1 to 'random' sent successfully.")
    else:
        print("Failed to send message 1 to 'random'.")

    print("\nCurrent chat rooms storage:")
    for room_id, messages in chat_rooms_storage.items():
        print(f"\nChat Room: {room_id}")
        for msg_entry in messages:
            print(f"  {msg_entry['sender']}: {msg_entry['message']}")
    
    # Example of a potential failure (though hard to simulate with list.append and dict creation)
    # For demonstration, let's imagine a scenario where it could fail.
    # For now, this will always succeed with valid inputs.
