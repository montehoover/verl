def display_employee_profile(employee_data, exclude_fields=None):
    """
    Formats employee data into a readable string, excluding specified fields.

    Args:
        employee_data (dict): A dictionary containing employee information.
                              Example: {'name': 'John Doe', 'position': 'Developer', ...}
        exclude_fields (list, optional): A list of field names (keys) to exclude
                                         from the output. Defaults to None, which
                                         means no fields are excluded by default.
                                         Example: ['social_security_number']

    Returns:
        str: A formatted string representing the employee's profile.
    """
    if exclude_fields is None:
        exclude_fields = []

    profile_parts = []
    for key, value in employee_data.items():
        if key not in exclude_fields:
            # Replace underscores with spaces and capitalize words for better readability
            formatted_key = key.replace('_', ' ').title()
            profile_parts.append(f"{formatted_key}: {value}")

    return "\n".join(profile_parts)

if __name__ == '__main__':
    # Example Usage
    sample_employee_data = {
        'name': 'Jane Doe',
        'position': 'Senior Software Engineer',
        'salary': 120000,
        'department': 'Technology',
        'email': 'jane.doe@example.com',
        'social_security_number': '***-**-1234',
        'employee_id': 'E1001'
    }

    print("--- Full Profile ---")
    print(display_employee_profile(sample_employee_data))
    print("\n--- Profile excluding SSN ---")
    print(display_employee_profile(sample_employee_data, exclude_fields=['social_security_number']))
    print("\n--- Profile excluding SSN and Salary ---")
    print(display_employee_profile(sample_employee_data, exclude_fields=['social_security_number', 'salary']))
    print("\n--- Profile with no exclusions (explicit empty list) ---")
    print(display_employee_profile(sample_employee_data, exclude_fields=[]))
