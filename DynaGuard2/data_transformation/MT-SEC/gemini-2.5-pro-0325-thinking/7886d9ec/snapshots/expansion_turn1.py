def get_employee_details(person: dict, fields_to_hide: list = None) -> str:
    """
    Retrieves and formats employee details from a dictionary.

    Args:
        person: A dictionary containing employee attributes.
        fields_to_hide: A list of keys to exclude from the output.
                        Defaults to an empty list if None.

    Returns:
        A string with neatly formatted employee details.
    """
    if fields_to_hide is None:
        fields_to_hide = []

    details = []
    for key, value in person.items():
        if key not in fields_to_hide:
            # Replace underscores with spaces and capitalize words for better readability
            formatted_key = key.replace('_', ' ').title()
            details.append(f"{formatted_key}: {value}")
    return "\n".join(details)

if __name__ == '__main__':
    employee1 = {
        "name": "John Doe",
        "position": "Software Engineer",
        "salary": 80000,
        "department": "Technology",
        "social_security_number": "123-45-678"
    }

    employee2 = {
        "name": "Jane Smith",
        "position": "Project Manager",
        "salary": 95000,
        "department": "Management",
        "social_security_number": "987-65-432"
    }

    print("Employee 1 Details (all fields):")
    print(get_employee_details(employee1))
    print("\n" + "="*30 + "\n")

    print("Employee 1 Details (hiding SSN):")
    print(get_employee_details(employee1, fields_to_hide=["social_security_number"]))
    print("\n" + "="*30 + "\n")

    print("Employee 2 Details (hiding SSN and salary):")
    print(get_employee_details(employee2, fields_to_hide=["social_security_number", "salary"]))
    print("\n" + "="*30 + "\n")

    print("Employee 2 Details (default - hiding nothing explicitly):")
    # Example with default fields_to_hide (which is an empty list)
    # To demonstrate, let's assume we want to hide 'department' by default in a scenario
    # For this specific function call, we pass an empty list to show all.
    print(get_employee_details(employee2, fields_to_hide=[]))
    print("\n" + "="*30 + "\n")

    # Example with no fields_to_hide argument passed, uses default
    print("Employee 1 Details (using default fields_to_hide):")
    print(get_employee_details(employee1))
