def display_employee_info(staff, exclude_sensitive=False):
    """
    Formats and displays employee details.

    Args:
        staff (dict): A dictionary containing employee information.
                      Expected keys: "name", "position", "salary",
                                     "department", "social_security_number".
        exclude_sensitive (bool): If True, sensitive information like
                                  social_security_number will be excluded.
                                  Defaults to False.

    Returns:
        str: A formatted string containing the employee's details.
    """
    details = []
    details.append(f"Name: {staff.get('name', 'N/A')}")
    details.append(f"Position: {staff.get('position', 'N/A')}")
    details.append(f"Salary: {staff.get('salary', 'N/A')}")
    details.append(f"Department: {staff.get('department', 'N/A')}")

    if not exclude_sensitive:
        details.append(f"Social Security Number: {staff.get('social_security_number', 'N/A')}")

    return "\n".join(details)


import logging
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EDITABLE_FIELDS = {"name", "position", "department"}

def update_and_log_employee(staff, changes):
    """
    Updates employee details and logs the changes.

    Args:
        staff (dict): The employee's current information.
        changes (dict): A dictionary of changes to apply.
                        Keys are attribute names, values are new values.

    Returns:
        dict: The updated employee dictionary.
    """
    employee_name = staff.get("name", "Unknown Employee")
    for key, new_value in changes.items():
        old_value = staff.get(key, 'N/A')
        if old_value != new_value:
            staff[key] = new_value
            logging.info(f"Employee '{employee_name}': Changed '{key}' from '{old_value}' to '{new_value}'")
        else:
            logging.info(f"Employee '{employee_name}': No change for '{key}', value already '{new_value}'")
    return staff

def modify_staff_info(staff, changes, admin_status):
    """
    Modifies employee information based on role-based restrictions.

    Args:
        staff (dict): The employee's current information.
        changes (dict): A dictionary of changes to apply.
        admin_status (bool): True if the user is an admin, False otherwise.

    Returns:
        dict: The updated employee dictionary.
    """
    employee_name = staff.get("name", "Unknown Employee")
    allowed_changes = {}

    if admin_status:
        allowed_changes = changes
    else:
        for key, value in changes.items():
            if key in EDITABLE_FIELDS:
                allowed_changes[key] = value
            else:
                logging.warning(
                    f"User (non-admin) attempted to change restricted field '{key}' for employee '{employee_name}'. "
                    f"Change not applied."
                )
    
    if not allowed_changes:
        logging.info(f"No authorized changes to apply for employee '{employee_name}'.")
        return staff

    return update_and_log_employee(staff, allowed_changes)

if __name__ == '__main__':
    employee1 = {
        "name": "John Doe",
        "position": "Software Engineer",
        "salary": "$90,000",
        "department": "Technology",
        "social_security_number": "XXX-XX-XXXX"
    }

    employee2 = {
        "name": "Jane Smith",
        "position": "Project Manager",
        "salary": "$110,000",
        "department": "Management",
        "social_security_number": "YYY-YY-YYYY"
    }

    print("Displaying all information for John Doe:")
    print(display_employee_info(employee1))
    print("\nDisplaying information for John Doe (excluding sensitive):")
    print(display_employee_info(employee1, exclude_sensitive=True))

    print("\nDisplaying all information for Jane Smith:")
    print(display_employee_info(employee2, exclude_sensitive=False))
    print("\nDisplaying information for Jane Smith (excluding sensitive):")
    print(display_employee_info(employee2, True))

    # Example with missing keys
    employee3_incomplete = {
        "name": "Alice Brown",
        "position": "Intern"
        # salary, department, social_security_number are missing
    }
    print("\nDisplaying information for Alice Brown (all info, some missing):")
    print(display_employee_info(employee3_incomplete))
    print("\nDisplaying information for Alice Brown (excluding sensitive, some missing):")
    print(display_employee_info(employee3_incomplete, exclude_sensitive=True))

    print("\n--- Testing update_and_log_employee ---")
    print("Original employee1 data:")
    print(display_employee_info(employee1))

    changes_to_employee1 = {
        "position": "Senior Software Engineer",
        "salary": "$100,000",
        "department": "Advanced Technology" # New department
    }
    print(f"\nUpdating employee1 with changes: {changes_to_employee1}")
    updated_employee1 = update_and_log_employee(employee1.copy(), changes_to_employee1) # Use .copy() to avoid modifying original dict directly in this example run
    print("\nUpdated employee1 data:")
    print(display_employee_info(updated_employee1))

    # Example of updating a non-existent key (it will be added)
    changes_add_key = {
        "start_date": "2023-01-15"
    }
    print(f"\nUpdating employee1 with new key: {changes_add_key}")
    updated_employee1_new_key = update_and_log_employee(updated_employee1.copy(), changes_add_key)
    print("\nUpdated employee1 data with new key:")
    print(display_employee_info(updated_employee1_new_key))

    # Example of no actual change
    no_actual_changes = {
        "name": "John Doe"
    }
    print(f"\nAttempting to update employee1 with no actual change: {no_actual_changes}")
    updated_employee1_no_change = update_and_log_employee(updated_employee1_new_key.copy(), no_actual_changes)
    print("\nEmployee1 data after attempting no change:")
    print(display_employee_info(updated_employee1_no_change))

    print("\n--- Testing modify_staff_info ---")
    # Reset employee2 for these tests
    employee2_original = {
        "name": "Jane Smith",
        "position": "Project Manager",
        "salary": "$110,000",
        "department": "Management",
        "social_security_number": "YYY-YY-YYYY"
    }

    # Test case 1: Admin user trying to change salary (allowed) and position (allowed)
    admin_changes = {"salary": "$120,000", "position": "Senior Project Manager"}
    print(f"\nAdmin attempting to modify employee2 with changes: {admin_changes}")
    employee2_admin_updated = modify_staff_info(employee2_original.copy(), admin_changes, admin_status=True)
    print("Employee2 data after admin update:")
    print(display_employee_info(employee2_admin_updated))

    # Test case 2: Non-admin user trying to change position (allowed) and salary (restricted)
    non_admin_changes = {"position": "Team Lead", "salary": "$115,000"}
    print(f"\nNon-admin attempting to modify employee2 with changes: {non_admin_changes}")
    employee2_non_admin_updated = modify_staff_info(employee2_original.copy(), non_admin_changes, admin_status=False)
    print("Employee2 data after non-admin update (salary change should be ignored):")
    print(display_employee_info(employee2_non_admin_updated))
    
    # Test case 3: Non-admin user trying to change only allowed fields
    non_admin_allowed_changes = {"department": "Operations", "name": "Jane A. Smith"}
    print(f"\nNon-admin attempting to modify employee2 with allowed changes: {non_admin_allowed_changes}")
    employee2_non_admin_allowed_updated = modify_staff_info(employee2_original.copy(), non_admin_allowed_changes, admin_status=False)
    print("Employee2 data after non-admin update with allowed fields:")
    print(display_employee_info(employee2_non_admin_allowed_updated))

    # Test case 4: Admin user trying to change SSN (allowed for admin)
    admin_ssn_change = {"social_security_number": "ZZZ-ZZ-ZZZZ"}
    print(f"\nAdmin attempting to modify employee2 SSN: {admin_ssn_change}")
    employee2_admin_ssn_updated = modify_staff_info(employee2_original.copy(), admin_ssn_change, admin_status=True)
    print("Employee2 data after admin SSN update:")
    print(display_employee_info(employee2_admin_ssn_updated)) # Display with sensitive info to check

    # Test case 5: Non-admin user trying to change only restricted fields
    non_admin_restricted_changes = {"salary": "$105,000", "social_security_number": "AAA-AA-AAAA"}
    print(f"\nNon-admin attempting to modify employee2 with only restricted changes: {non_admin_restricted_changes}")
    employee2_non_admin_restricted_updated = modify_staff_info(employee2_original.copy(), non_admin_restricted_changes, admin_status=False)
    print("Employee2 data after non-admin attempt to change only restricted fields (no changes should apply):")
    print(display_employee_info(employee2_non_admin_restricted_updated))
