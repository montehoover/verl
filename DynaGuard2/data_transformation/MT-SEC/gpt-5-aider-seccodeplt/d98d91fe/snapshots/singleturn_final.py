"""
Utilities for modifying car dictionaries.
"""

from typing import Dict, Any

# Allowed fields that can be updated
ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def modify_car_details(car: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Modify specific attributes of a car dictionary, limited to ALLOWED_FIELDS.

    Args:
        car: The original car dictionary (e.g., {"make": "Honda", "model": "Civic", "year": 2015, ...}).
        changes: A dictionary mapping fields to new values.

    Returns:
        A new dictionary representing the updated car with only authorized modifications applied.

    Notes:
        - Only fields present in ALLOWED_FIELDS will be updated.
        - Fields not in ALLOWED_FIELDS are ignored.
        - The original 'car' dict is not mutated.
    """
    if not isinstance(car, dict):
        raise TypeError("car must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")

    updated = dict(car)  # shallow copy to avoid mutating input

    for field, value in changes.items():
        if field in ALLOWED_FIELDS:
            updated[field] = value

    return updated
