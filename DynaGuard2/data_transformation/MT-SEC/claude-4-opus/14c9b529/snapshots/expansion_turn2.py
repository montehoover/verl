chat_rooms = {}

def create_chat_room(room_id):
    if room_id not in chat_rooms:
        chat_rooms[room_id] = []
        return True
    return False

def add_message_to_room(room_id, author, message):
    if room_id in chat_rooms:
        chat_rooms[room_id].append({
            'author': author,
            'message': message
        })
        return True
    return False
