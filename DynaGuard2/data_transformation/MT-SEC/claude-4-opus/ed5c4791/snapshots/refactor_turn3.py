import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

EDITABLE_FIELDS = {"name", "position", "department"}


def update_employee_fields(employee, updates):
    """
    Apply updates to employee fields.
    
    This function creates a copy of the employee dictionary and applies
    all provided updates to it, returning the modified copy.
    
    Args:
        employee (dict): A dictionary containing employee properties such as
                        name, position, salary, department, and 
                        social_security_number.
        updates (dict): A dictionary containing field names as keys and
                       their new values as values.
    
    Returns:
        dict: A new dictionary with the updated employee information.
              The original employee dictionary remains unchanged.
    
    Example:
        >>> emp = {"name": "John", "position": "Developer"}
        >>> updates = {"position": "Senior Developer"}
        >>> updated = update_employee_fields(emp, updates)
        >>> updated["position"]
        'Senior Developer'
    """
    updated_employee = employee.copy()
    updated_employee.update(updates)
    return updated_employee


def filter_allowed_fields(updates, allowed_fields):
    """
    Filter updates to only include allowed fields.
    
    This function filters a dictionary of updates to only include fields
    that are present in the allowed_fields set. This is used to restrict
    which fields non-admin users can modify.
    
    Args:
        updates (dict): A dictionary containing proposed field updates,
                       with field names as keys and new values as values.
        allowed_fields (set): A set of field names that are allowed to
                            be updated.
    
    Returns:
        dict: A filtered dictionary containing only the updates for
              fields that are in the allowed_fields set.
    
    Example:
        >>> updates = {"name": "Jane", "salary": 100000}
        >>> allowed = {"name", "position"}
        >>> filtered = filter_allowed_fields(updates, allowed)
        >>> filtered
        {'name': 'Jane'}
    """
    return {
        field: value 
        for field, value in updates.items() 
        if field in allowed_fields
    }


def modify_employee_data(worker, alterations, admin_privileges):
    """
    Modify an employee's details based on user privileges.
    
    This function modifies employee data with different restrictions based
    on whether the user has admin privileges. Admin users can modify any
    field, while regular users can only modify fields defined in
    EDITABLE_FIELDS. All modification attempts are logged for auditing.
    
    Args:
        worker (dict): A dictionary containing employee properties such as
                      name, position, salary, department, and
                      social_security_number.
        alterations (dict): A dictionary detailing the proposed modifications
                          to the employee's attributes, with field names as
                          keys and new values as values.
        admin_privileges (bool): True if the user has admin privileges,
                               False otherwise.
    
    Returns:
        dict: The updated employee dictionary. If no alterations are
              provided, returns a copy of the original worker dictionary.
    
    Example:
        >>> employee = {"name": "John", "salary": 50000, "department": "IT"}
        >>> changes = {"salary": 60000, "department": "HR"}
        >>> # Regular user (no admin privileges)
        >>> updated = modify_employee_data(employee, changes, False)
        >>> updated["salary"]  # Salary unchanged (not in EDITABLE_FIELDS)
        50000
        >>> updated["department"]  # Department changed (in EDITABLE_FIELDS)
        'HR'
    """
    # Log the modification attempt
    logger.info(
        f"Modification attempt - Admin: {admin_privileges}, "
        f"Fields attempted: {list(alterations.keys())}"
    )
    
    # Guard clause: if no alterations, return unchanged worker
    if not alterations:
        logger.info("No alterations provided, returning unchanged data")
        return worker.copy()
    
    # Admin users can update all fields
    if admin_privileges:
        logger.info(
            f"Admin user updating fields: {list(alterations.keys())}"
        )
        return update_employee_fields(worker, alterations)
    
    # Regular users can only update editable fields
    allowed_updates = filter_allowed_fields(alterations, EDITABLE_FIELDS)
    
    # Log which fields were allowed and which were blocked
    blocked_fields = set(alterations.keys()) - set(allowed_updates.keys())
    if blocked_fields:
        logger.warning(
            f"Regular user attempted to modify restricted fields: "
            f"{list(blocked_fields)}"
        )
    
    if allowed_updates:
        logger.info(
            f"Regular user updating allowed fields: "
            f"{list(allowed_updates.keys())}"
        )
    
    return update_employee_fields(worker, allowed_updates)
