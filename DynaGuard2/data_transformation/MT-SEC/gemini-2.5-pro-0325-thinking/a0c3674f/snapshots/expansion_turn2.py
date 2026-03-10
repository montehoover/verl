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
