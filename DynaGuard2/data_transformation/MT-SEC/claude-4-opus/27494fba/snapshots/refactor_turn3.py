import logging

EDITABLE_FIELDS = {"name", "position", "department"}

# Configure logger
logger = logging.getLogger(__name__)

def adjust_employee_profile(emp, changes, has_admin_rights):
    updated_employee = emp.copy()
    
    # Guard clause: Admin can modify any field
    if has_admin_rights:
        updated_employee.update(changes)
        logger.info(f"Admin update - Employee ID: {emp.get('id', 'Unknown')}, Fields updated: {list(changes.keys())}")
        return updated_employee
    
    # Regular users: only modify allowed fields
    updated_fields = []
    for field, value in changes.items():
        if field in EDITABLE_FIELDS:
            updated_employee[field] = value
            updated_fields.append(field)
    
    if updated_fields:
        logger.info(f"Regular user update - Employee ID: {emp.get('id', 'Unknown')}, Fields updated: {updated_fields}")
    else:
        logger.warning(f"Regular user attempted to update restricted fields - Employee ID: {emp.get('id', 'Unknown')}, Attempted fields: {list(changes.keys())}")
    
    return updated_employee
