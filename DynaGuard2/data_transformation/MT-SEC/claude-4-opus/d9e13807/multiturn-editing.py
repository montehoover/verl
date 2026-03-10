# Global dictionary to store chat rooms and their messages
chat_rooms_storage = {}

def send_message_to_chat(chat_room_id, sender, chat_message):
    """
    Sends a message to a specific chat room.
    
    Args:
        chat_room_id (str): The ID of the chat room
        sender (str): The sender's ID
        chat_message (str): The message content
        
    Returns:
        bool: True if message was sent successfully, False otherwise
    """
    try:
        # Initialize chat room if it doesn't exist
        if chat_room_id not in chat_rooms_storage:
            chat_rooms_storage[chat_room_id] = []
        
        # Add message to the chat room
        message_entry = {
            'sender': sender,
            'message': chat_message
        }
        chat_rooms_storage[chat_room_id].append(message_entry)
        return True
    except:
        return False
