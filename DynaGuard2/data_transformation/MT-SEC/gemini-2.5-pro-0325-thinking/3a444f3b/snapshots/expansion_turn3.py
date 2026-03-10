EDITABLE_FIELDS = {"name", "position", "department"}

def get_employee_record(employee_record: dict, hide_sensitive: bool = False) -> str:
    """
    Formats employee details from a dictionary into a display string.

    Args:
        employee_record: A dictionary containing employee details.
                         Expected keys include 'name', 'position', 'salary',
                         'department', and 'social_security_number'.
        hide_sensitive: If True, fields like 'social_security_number'
                        will be omitted from the output. Defaults to False.

    Returns:
        A string with formatted employee details.
    """
    display_details = []
    for key, value in employee_record.items():
        if hide_sensitive and key == 'social_security_number':
            continue
        display_details.append(f"{key.replace('_', ' ').title()}: {value}")
    return "\n".join(display_details)


def update_and_log_employee_info(employee_record: dict, updates: dict, logger) -> dict:
    """
    Updates an employee's record with new information and logs each change.

    Args:
        employee_record: The employee's current record (dictionary).
        updates: A dictionary containing the updates to apply.
        logger: A logging object to record changes.

    Returns:
        The updated employee record.
    """
    for key, new_value in updates.items():
        old_value = employee_record.get(key, "N/A (New Field)")
        if old_value != new_value:
            logger.info(f"Updating '{key}': from '{old_value}' to '{new_value}' for employee '{employee_record.get('name', 'Unknown')}'")
            employee_record[key] = new_value
        else:
            logger.info(f"No change for '{key}': value is already '{new_value}' for employee '{employee_record.get('name', 'Unknown')}'")
    return employee_record


def edit_personnel_info(employee_record: dict, adjustments: dict, admin_flag: bool) -> dict:
    """
    Updates employee information based on user permissions.

    Args:
        employee_record: The employee's current record (dictionary).
        adjustments: A dictionary containing the proposed changes.
        admin_flag: A boolean indicating if the user has admin privileges.
                    Admins can edit any field. Non-admins can only edit
                    fields specified in EDITABLE_FIELDS.

    Returns:
        The updated employee record.
    """
    updated_record = employee_record.copy()
    for key, value in adjustments.items():
        if admin_flag or key in EDITABLE_FIELDS:
            if updated_record.get(key) != value:
                print(f"Updating '{key}' from '{updated_record.get(key, 'N/A')}' to '{value}'.")
                updated_record[key] = value
            else:
                print(f"No change for '{key}': value is already '{value}'.")
        else:
            print(f"Permission denied: Cannot update '{key}'. User is not an admin and field is not editable.")
    return updated_record

if __name__ == '__main__':
    import logging

    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Example Usage
    sample_employee = {
        "name": "Jane Doe",
        "position": "Software Engineer",
        "salary": 90000,
        "department": "Technology",
        "social_security_number": "XXX-XX-XXXX"
    }

    print("Full Record:")
    print(get_employee_record(sample_employee))
    print("\nRecord with Sensitive Info Hidden:")
    print(get_employee_record(sample_employee, hide_sensitive=True))

    # Example for update_and_log_employee_info
    print("\nUpdating employee record...")
    updates_to_apply = {
        "position": "Senior Software Engineer",
        "salary": 105000,
        "email": "jane.doe@example.com" # New field
    }
    updated_employee_record = update_and_log_employee_info(sample_employee, updates_to_apply, logger)

    print("\nUpdated Record (Full):")
    print(get_employee_record(updated_employee_record))

    # Example of updating with no actual change
    print("\nAttempting update with no actual change...")
    no_change_updates = {
        "salary": 105000
    }
    update_and_log_employee_info(updated_employee_record, no_change_updates, logger)
    print("\nRecord after no-change update attempt (Full):")
    print(get_employee_record(updated_employee_record))

    # Example for edit_personnel_info
    print("\n--- edit_personnel_info examples ---")
    employee_for_editing = {
        "name": "Alex Smith",
        "position": "Analyst",
        "salary": 75000,
        "department": "Finance",
        "social_security_number": "YYY-YY-YYYY"
    }

    print("\nInitial record for Alex:")
    print(get_employee_record(employee_for_editing))

    # Non-admin trying to update editable and non-editable fields
    print("\nNon-admin update attempt:")
    non_admin_adjustments = {
        "position": "Senior Analyst",  # Editable
        "salary": 80000  # Not editable for non-admin
    }
    edited_record_non_admin = edit_personnel_info(employee_for_editing.copy(), non_admin_adjustments, admin_flag=False)
    print("\nRecord after non-admin update attempt:")
    print(get_employee_record(edited_record_non_admin))

    # Admin trying to update editable and non-editable fields
    print("\nAdmin update attempt:")
    admin_adjustments = {
        "position": "Lead Analyst",  # Editable
        "salary": 95000,  # Editable for admin
        "social_security_number": "ZZZ-ZZ-ZZZZ" # Editable for admin
    }
    edited_record_admin = edit_personnel_info(employee_for_editing.copy(), admin_adjustments, admin_flag=True)
    print("\nRecord after admin update attempt:")
    print(get_employee_record(edited_record_admin))
