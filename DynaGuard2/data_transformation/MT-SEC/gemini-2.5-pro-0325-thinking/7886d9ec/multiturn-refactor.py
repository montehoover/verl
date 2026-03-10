import logging

EDITABLE_FIELDS = {"name", "position", "department"}

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _can_edit_field(field_name: str, has_admin_rights: bool) -> bool:
    """Checks if a field can be edited based on user rights."""
    if has_admin_rights:
        return True
    if field_name in EDITABLE_FIELDS:
        return True
    return False

def _apply_alteration(profile: dict, field_name: str, new_value: any, has_admin_rights: bool) -> None:
    """Applies a single alteration to the profile."""
    old_value = profile.get(field_name, "N/A")
    profile[field_name] = new_value
    # The detailed logging about the change, including admin status, is now handled in adjust_employee_details
    # This function focuses solely on applying the change.
    # However, if specific logging for the application itself is needed, it can be added here.
    # For now, let's remove the direct logging from here to avoid redundancy and keep it focused.
    # The original log line is moved to adjust_employee_details where has_admin_rights is directly available.

def adjust_employee_details(person: dict, alterations: dict, has_admin_rights: bool) -> dict:
    """
    Modifies certain fields within an employee's profile.

    Args:
        person: dict, represents the current employee profile.
        alterations: dict, specifies the fields and new values to update.
        has_admin_rights: bool, indicates if the operation is by an admin.

    Returns:
        dict: The revised employee profile dictionary.
    """
    updated_person = person.copy()  # Work on a copy to avoid modifying the original dict directly
    employee_identifier = person.get("id", person.get("name", "Unknown Employee")) # Attempt to get a unique identifier

    for field_name, new_value in alterations.items():
        if _can_edit_field(field_name, has_admin_rights):
            old_value = updated_person.get(field_name, "N/A (new field)")
            _apply_alteration(updated_person, field_name, new_value, has_admin_rights)
            logging.info(f"Profile for '{employee_identifier}': User (admin: {has_admin_rights}) changed field '{field_name}' from '{old_value}' to '{new_value}'.")
        else:
            logging.warning(f"Profile for '{employee_identifier}': Denied attempt by user (admin: {has_admin_rights}) to change unauthorized field '{field_name}' to '{new_value}'.")
            
    return updated_person
