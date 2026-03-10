chat_rooms = {}

def check_user_in_room(room_name: str, member_id: str) -> bool:
    """
    Return True if member_id is present in the chat room identified by room_name, otherwise False.

    The global chat_rooms dictionary maps room names to a collection of associated users.
    This function supports common collection types:
    - set, list, tuple: membership directly checked
    - dict: membership checked against keys
    - other containers (excluding str/bytes): membership attempted via `in`
    """
    members = chat_rooms.get(room_name)
    if members is None:
        return False

    if isinstance(members, (set, list, tuple)):
        return member_id in members

    if isinstance(members, dict):
        return member_id in members

    if hasattr(members, "__contains__") and not isinstance(members, (str, bytes)):
        try:
            return member_id in members
        except Exception:
            return False

    return False
