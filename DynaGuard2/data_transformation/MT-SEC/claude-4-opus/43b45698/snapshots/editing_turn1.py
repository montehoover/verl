chat_rooms = {}

def check_user_in_room(room_name, member_id):
    if room_name in chat_rooms:
        return member_id in chat_rooms[room_name]
    return False
