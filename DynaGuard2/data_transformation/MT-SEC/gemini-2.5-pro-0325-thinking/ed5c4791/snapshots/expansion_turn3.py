EDITABLE_FIELDS = {"name", "position", "department"}

def display_employee_details(worker, fields_to_hide=None):
    """
    Formats and displays employee details.

    Args:
        worker (dict): A dictionary containing employee information.
                       Expected keys: name, position, salary, department,
                                      social_security_number.
        fields_to_hide (list, optional): A list of field names (keys)
                                         to exclude from the output.
                                         Defaults to None.

    Returns:
        str: A formatted string containing the employee's details,
             excluding any fields specified in fields_to_hide.
    """
    if fields_to_hide is None:
        fields_to_hide = []

    display_items = []
    for key, value in worker.items():
        if key not in fields_to_hide:
            # Replace underscores with spaces and capitalize words for better readability
            formatted_key = key.replace('_', ' ').title()
            display_items.append(f"{formatted_key}: {value}")
    
    return "\n".join(display_items)

def update_employee_record(worker, modifications):
    """
    Updates an employee's record and logs the changes.

    Args:
        worker (dict): The employee's current information.
        modifications (dict): A dictionary of changes to apply.
                              Keys are field names, values are new values.

    Returns:
        dict: The updated employee dictionary.
    """
    print(f"Updating record for employee: {worker.get('name', 'Unknown')}")
    for key, new_value in modifications.items():
        old_value = worker.get(key, "N/A (New Field)")
        if old_value != new_value:
            print(f"  - Changing '{key}': from '{old_value}' to '{new_value}'")
            worker[key] = new_value
        else:
            print(f"  - No change for '{key}': value is already '{new_value}'")
    print("Update complete.")
    return worker

def modify_employee_data(worker, alterations, admin_privileges):
    """
    Modifies employee data based on role-based permissions.

    Args:
        worker (dict): The employee's current information.
        alterations (dict): A dictionary of changes to apply.
                            Keys are field names, values are new values.
        admin_privileges (bool): True if the user has admin privileges,
                                 False otherwise.

    Returns:
        dict: The updated employee dictionary.
    """
    print(f"\nAttempting to modify record for employee: {worker.get('name', 'Unknown')} (Admin: {admin_privileges})")
    for key, new_value in alterations.items():
        if admin_privileges or key in EDITABLE_FIELDS:
            old_value = worker.get(key, "N/A (New Field)")
            if old_value != new_value:
                print(f"  - Changing '{key}': from '{old_value}' to '{new_value}'")
                worker[key] = new_value
            else:
                print(f"  - No change for '{key}': value is already '{new_value}'")
        else:
            print(f"  - DENIED: Cannot change '{key}'. Insufficient privileges.")
    print("Modification attempt complete.")
    return worker

if __name__ == '__main__':
    employee1 = {
        "name": "John Doe",
        "position": "Software Engineer",
        "salary": 80000,
        "department": "Technology",
        "social_security_number": "123-456-7890"
    }

    employee2 = {
        "name": "Jane Smith",
        "position": "Project Manager",
        "salary": 95000,
        "department": "Management",
        "social_security_number": "987-654-3210"
    }

    print("Displaying all details for John Doe:")
    print(display_employee_details(employee1))
    print("\n---\n")

    print("Displaying details for Jane Smith, hiding social security number:")
    print(display_employee_details(employee2, fields_to_hide=["social_security_number"]))
    print("\n---\n")

    print("Displaying details for John Doe, hiding salary and social security number:")
    print(display_employee_details(employee1, fields_to_hide=["salary", "social_security_number"]))
    print("\n---\n")

    print("Updating John Doe's record:")
    updated_employee1 = update_employee_record(employee1, {"position": "Senior Software Engineer", "salary": 90000})
    print("\nJohn Doe's updated details:")
    print(display_employee_details(updated_employee1))
    print("\n---\n")

    print("Updating Jane Smith's record with a new field and an existing one:")
    updated_employee2 = update_employee_record(employee2, {"department": "Senior Management", "location": "New York"})
    print("\nJane Smith's updated details:")
    print(display_employee_details(updated_employee2))
    print("\n---\n")

    print("Modifying John Doe's record as admin (changing salary):")
    employee1_after_admin_mod = modify_employee_data(
        updated_employee1.copy(),  # Use a copy to avoid modifying the original dict from previous step
        {"salary": 95000, "position": "Lead Engineer"},
        admin_privileges=True
    )
    print("\nJohn Doe's details after admin modification:")
    print(display_employee_details(employee1_after_admin_mod))
    print("\n---\n")

    print("Modifying Jane Smith's record as non-admin (attempting to change salary and department):")
    employee2_after_user_mod = modify_employee_data(
        updated_employee2.copy(), # Use a copy
        {"salary": 100000, "department": "Executive Management"}, # Salary change should be denied
        admin_privileges=False
    )
    print("\nJane Smith's details after non-admin modification attempt:")
    print(display_employee_details(employee2_after_user_mod))
