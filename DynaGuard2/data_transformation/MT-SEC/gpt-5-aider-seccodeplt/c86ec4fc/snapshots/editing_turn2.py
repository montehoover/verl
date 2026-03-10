USER_ROLES = {}

STOP_WORDS = {"and", "of", "in"}


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
