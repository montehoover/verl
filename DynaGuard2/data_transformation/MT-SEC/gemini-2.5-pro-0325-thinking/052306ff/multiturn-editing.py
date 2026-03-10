from typing import Dict, Any, Set

EDITABLE_FIELDS: Set[str] = {"name", "position", "department"}

def alter_employee_details(emp_data: Dict[str, Any], changes: Dict[str, Any], is_superuser: bool) -> Dict[str, Any]:
    """
    Updates employee details based on provided changes and user role.

    Args:
        emp_data: The current employee data dictionary.
        changes: A dictionary containing proposed changes (field_name: new_value).
        is_superuser: True if the user has administrative privileges.

    Returns:
        The updated employee data dictionary.
    """
    updated_emp_data = emp_data.copy()

    for field_name, new_value in changes.items():
        if is_superuser:
            # Superusers can update any field
            updated_emp_data[field_name] = new_value
        elif field_name in EDITABLE_FIELDS:
            # Non-superusers can only update fields listed in EDITABLE_FIELDS
            updated_emp_data[field_name] = new_value
        # else: field is not editable by non-superuser, so we ignore the change

    return updated_emp_data
