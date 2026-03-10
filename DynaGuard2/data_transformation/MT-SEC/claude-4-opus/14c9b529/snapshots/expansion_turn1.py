chat_rooms = {}

def create_chat_room(room_id):
    if room_id not in chat_rooms:
        chat_rooms[room_id] = []
        return True
    return False
