# Global dictionary to store chat messages for multiple rooms
chat_rooms = {}

def insert_chat_message(chat_room_id, author_id, message_content):
    """
    Inserts a message into a specific chat room.
    
    Args:
        chat_room_id (str): The ID of the chat room
        author_id (str): The ID of the message author
        message_content (str): The content of the message
    
    Returns:
        bool: True if message is successfully added, False otherwise
    """
    try:
        # Create room if it doesn't exist
        if chat_room_id not in chat_rooms:
            chat_rooms[chat_room_id] = []
        
        message = {
            'author_id': author_id,
            'message_content': message_content
        }
        chat_rooms[chat_room_id].append(message)
        return True
    except:
        return False
