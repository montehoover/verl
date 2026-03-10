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

if __name__ == '__main__':
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
