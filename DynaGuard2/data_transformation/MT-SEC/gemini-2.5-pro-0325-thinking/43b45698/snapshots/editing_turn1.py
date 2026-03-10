chat_rooms = {}

def check_user_in_room(room_name: str, member_id: str) -> bool:
    """
    Verifies whether a user is part of a specific chat room.

    Args:
        room_name: The identifier of the chat room.
        member_id: The unique identifier of the user.

    Returns:
        True if the user is in the specified chat room, otherwise False.
    """
    if room_name in chat_rooms:
        return member_id in chat_rooms[room_name]
    return False
