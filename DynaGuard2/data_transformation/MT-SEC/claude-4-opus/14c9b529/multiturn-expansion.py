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

def log_chat_message(target_room, author_identifier, text_message):
    if target_room in chat_rooms:
        chat_rooms[target_room].append({
            'author': author_identifier,
            'message': text_message
        })
        return True
    return False
