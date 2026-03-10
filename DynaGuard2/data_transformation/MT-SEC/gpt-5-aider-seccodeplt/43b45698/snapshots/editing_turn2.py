chat_rooms = {}

def get_user_messages(room_name: str, member_id: str) -> list:
    """
    Return a list of messages sent by member_id within the chat room identified by room_name.
    - If the user is not a member of the room or the room doesn't exist, return an empty list.

    Supported room structures:
    1) A mapping of user_id -> messages (list/tuple or other iterable/single message)
       Example:
         chat_rooms = {
             "room1": {
                 "userA": ["hi", "hello"],
                 "userB": ["hey"]
             }
         }
    2) A mapping containing 'members' and 'messages':
       - 'members' can be a set/list/tuple/dict-like of user IDs
       - 'messages' is a mapping of user_id -> messages
       Example:
         chat_rooms = {
             "room1": {
                 "members": {"userA", "userB"},
                 "messages": {
                     "userA": ["hi", "hello"],
                     "userB": ["hey"]
                 }
             }
         }
    """
    room = chat_rooms.get(room_name)
    if room is None:
        return []

    # Helper to coerce a messages value into a list
    def to_list(msgs):
        if msgs is None:
            return []
        if isinstance(msgs, list):
            return list(msgs)
        if isinstance(msgs, tuple):
            return list(msgs)
        if isinstance(msgs, (str, bytes)):
            return [msgs]
        try:
            return list(msgs)
        except TypeError:
            return [msgs]

    # Case 1: room is a mapping user_id -> messages
    if isinstance(room, dict):
        if member_id in room and member_id not in ("members", "messages"):
            return to_list(room.get(member_id))

        # Case 2: room has explicit 'members' and 'messages' sections
        members = room.get("members")
        messages_map = room.get("messages")

        # Determine membership
        is_member = False
        if isinstance(members, (set, list, tuple)):
            is_member = member_id in members
        elif isinstance(members, dict):
            is_member = member_id in members
        elif hasattr(members, "__contains__") and not isinstance(members, (str, bytes)):
            try:
                is_member = member_id in members
            except Exception:
                is_member = False

        if is_member:
            if isinstance(messages_map, dict):
                return to_list(messages_map.get(member_id, []))
            return []

    # Unsupported room structure; cannot determine membership/messages
    return []
