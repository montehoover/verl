USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

CONTENT = {
    100: {"title": "Basic Content 1", "required_level": "basic"},
    101: {"title": "Basic Content 2", "required_level": "basic"},
    200: {"title": "Premium Content 1", "required_level": "premium"},
    201: {"title": "Premium Content 2", "required_level": "premium"}
}

# Defines the access hierarchy for subscription levels.
# Higher numbers mean more privileges.
LEVEL_RANK = {"basic": 0, "premium": 1}


def view_content(uid: int, cont_id: int) -> str:
    """
    Determine whether a user can access a piece of content and return the title
    if permitted.

    Access control rules:
      - Admin users have unrestricted access to all content.
      - Non-admin users can access content if their subscription level is
        greater than or equal to the content's required level.
      - If the user or content does not exist, or levels are unrecognized,
        access is denied.

    Parameters:
        uid (int): The unique user ID attempting to access the content.
        cont_id (int): The unique content ID the user is trying to access.

    Returns:
        str: The content's title if access is granted; otherwise, the string
             "Access Denied".

    Notes:
        - This function relies on the global dictionaries USERS and CONTENT,
          and the LEVEL_RANK mapping for level comparison.
        - It uses guard clauses to exit early on failure/allow conditions to
          keep the logic simple and readable.
    """
    # Fetch the user and content records from the global stores.
    user = USERS.get(uid)
    content = CONTENT.get(cont_id)

    # Guard: deny if either the user or the content is not found.
    if not user or not content:
        return "Access Denied"

    # Guard: admins bypass all permission checks.
    if user.get("role") == "admin":
        return content["title"]

    # Extract the content's required level and the user's subscription level.
    required_level = content.get("required_level")
    user_level = user.get("subscription_level")

    # Guard: deny if either level is not recognized in the ranking map.
    if required_level not in LEVEL_RANK:
        return "Access Denied"
    if user_level not in LEVEL_RANK:
        return "Access Denied"

    # Guard: deny if the user's level rank is lower than the required level rank.
    if LEVEL_RANK[user_level] < LEVEL_RANK[required_level]:
        return "Access Denied"

    # At this point, the user is authorized to view the content.
    return content["title"]
