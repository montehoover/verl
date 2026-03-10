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
