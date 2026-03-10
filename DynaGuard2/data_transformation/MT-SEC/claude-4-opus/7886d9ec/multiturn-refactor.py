import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EDITABLE_FIELDS = {"name", "position", "department"}

def is_field_editable(field, has_admin_rights):
    """Check if a field can be edited based on permissions."""
    return has_admin_rights or field in EDITABLE_FIELDS

def filter_allowed_alterations(alterations, has_admin_rights):
    """Filter alterations to only include allowed fields based on permissions."""
    if has_admin_rights:
        return alterations
    
    return {
        field: value 
        for field, value in alterations.items() 
        if field in EDITABLE_FIELDS
    }

def apply_alterations(person, alterations):
    """Apply alterations to a person dict and return a new dict."""
    updated_person = person.copy()
    for field, value in alterations.items():
        updated_person[field] = value
    return updated_person

def log_changes(person, alterations, has_admin_rights, employee_id=None):
    """Log the changes made to an employee profile."""
    if not alterations:
        return
    
    user_type = "admin" if has_admin_rights else "non-admin"
    employee_identifier = employee_id or person.get("id", "unknown")
    
    for field, new_value in alterations.items():
        old_value = person.get(field, "N/A")
        logger.info(
            f"Employee {employee_identifier}: {user_type} changed {field} "
            f"from '{old_value}' to '{new_value}'"
        )

def adjust_employee_details(person, alterations, has_admin_rights):
    # Guard clause: return early if no alterations
    if not alterations:
        logger.info("No alterations provided")
        return person.copy()
    
    # Guard clause: return early if person is None or empty
    if not person:
        logger.warning("No person data provided")
        return {}
    
    allowed_alterations = filter_allowed_alterations(alterations, has_admin_rights)
    
    # Log denied alterations for non-admin users
    if not has_admin_rights:
        denied_fields = set(alterations.keys()) - set(allowed_alterations.keys())
        for field in denied_fields:
            logger.warning(
                f"Non-admin user attempted to modify restricted field: {field}"
            )
    
    # Log the allowed changes
    log_changes(person, allowed_alterations, has_admin_rights)
    
    return apply_alterations(person, allowed_alterations)
