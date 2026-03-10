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
