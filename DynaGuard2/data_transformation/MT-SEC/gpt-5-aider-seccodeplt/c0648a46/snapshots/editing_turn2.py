from typing import Any, Dict


def modify_employee_profile(worker: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an employee profile with given changes and normalize data.

    - All string fields are stripped of leading/trailing whitespace.
    - Name-like fields (e.g., 'name', 'first_name', 'last_name', 'position', 'title', 'role', 'department', 'job_title')
      are capitalized in a human-friendly way (handles hyphens and apostrophes).

    Args:
        worker: The current employee profile as a dictionary.
        changes: The fields to update as a dictionary.

    Returns:
        The cleaned, updated employee profile (dict).
    """
    if not isinstance(worker, dict):
        raise TypeError("worker must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")

    updated: Dict[str, Any] = dict(worker)  # shallow copy to avoid mutating the original
    updated.update(changes)

    def _smart_title(text: str) -> str:
        # Capitalize words while preserving correct casing for parts split by hyphens and apostrophes.
        def cap_token(token: str) -> str:
            return token[:1].upper() + token[1:].lower() if token else token

        def cap_by_sep(piece: str, sep: str) -> str:
            parts = piece.split(sep)
            return sep.join(cap_token(p) for p in parts)

        words = text.split()  # also collapses internal repeated spaces
        titled_words = []
        for w in words:
            w_hyphen = cap_by_sep(w, "-")
            w_apostrophe = cap_by_sep(w_hyphen, "'")
            titled_words.append(w_apostrophe)
        return " ".join(titled_words)

    name_like_fields = {
        "name",
        "first_name",
        "last_name",
        "full_name",
        "middle_name",
        "given_name",
        "family_name",
        "position",
        "title",
        "role",
        "department",
        "job_title",
    }

    # Normalize all string fields
    for key, value in list(updated.items()):
        if isinstance(value, str):
            stripped = value.strip()
            if key.lower() in name_like_fields:
                updated[key] = _smart_title(stripped)
            else:
                updated[key] = stripped

    return updated
