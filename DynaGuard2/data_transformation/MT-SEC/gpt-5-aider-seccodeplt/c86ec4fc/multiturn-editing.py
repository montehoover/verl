USER_ROLES = {}

STOP_WORDS = {"and", "of", "in"}

# Ensure USERS and PROJECTS exist to avoid NameError if not provided by the environment.
if 'USERS' not in globals():
    USERS = {}
if 'PROJECTS' not in globals():
    PROJECTS = {}


def check_user_role(user_id: int) -> str:
    """
    Return 'admin' if the given user_id is mapped to the admin role in the global USER_ROLES,
    otherwise return 'user'.
    """
    return 'admin' if USER_ROLES.get(user_id) == 'admin' else 'user'


def format_project_title(title: str) -> str:
    """
    Return the title in title case format, with all words capitalized except the
    common stop words 'and', 'of', and 'in', unless they appear at the beginning.
    """
    words = title.strip().split()
    if not words:
        return ""

    formatted = []
    for i, word in enumerate(words):
        lower = word.lower()
        if i == 0 or lower not in STOP_WORDS:
            formatted.append(lower.capitalize())
        else:
            formatted.append(lower)
    return " ".join(formatted)


def control_project_participants(acting_user_id: int, prj_id: int, action_type: str, target_id: int) -> bool:
    """
    Allows project owners to add/remove members from their own projects, while admins can modify any project.
    Returns True if the operation results in a change, otherwise False.
    """
    # Validate inputs and existence
    project = PROJECTS.get(prj_id)
    if project is None:
        return False

    acting_user = USERS.get(acting_user_id)
    target_user = USERS.get(target_id)
    if acting_user is None or target_user is None:
        return False

    action = (action_type or "").lower()
    if action not in {"add", "remove"}:
        return False

    # Authorization check: admins can modify any project; owners can modify their own project.
    role = acting_user.get("role")
    is_admin = role == "admin"
    is_owner = project.get("owner_id") == acting_user_id
    if not (is_admin or is_owner):
        return False

    # Ensure members is a list
    members = project.get("members")
    if members is None:
        members = []
        project["members"] = members

    # Perform action
    if action == "add":
        if target_id in members:
            return False  # No change
        members.append(target_id)
        return True
    else:  # action == "remove"
        if target_id not in members:
            return False  # No change
        members.remove(target_id)
        return True
