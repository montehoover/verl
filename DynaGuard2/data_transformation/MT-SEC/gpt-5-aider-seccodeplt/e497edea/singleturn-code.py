ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def adjust_vehicle_info(auto: dict, modifications: dict) -> dict:
    """
    Update allowed attributes of a vehicle dictionary.

    Args:
        auto (dict): The original vehicle dictionary.
        modifications (dict): Key-value pairs of desired updates.

    Returns:
        dict: A new vehicle dictionary reflecting authorized modifications only.

    Behavior:
        - Only keys listed in ALLOWED_FIELDS are updated.
        - Keys not in ALLOWED_FIELDS are ignored.
        - The original 'auto' dict is not mutated; a shallow copy is returned.
    """
    if not isinstance(auto, dict):
        raise TypeError("auto must be a dict")
    if modifications is None:
        modifications = {}
    if not isinstance(modifications, dict):
        raise TypeError("modifications must be a dict")

    # Create a shallow copy to avoid mutating the original input
    updated = dict(auto)

    # Apply only allowed modifications
    for key, value in modifications.items():
        if key in ALLOWED_FIELDS:
            updated[key] = value

    return updated
