EDITABLE_FIELDS = {"name", "position", "department"}

def modify_employee_profile(worker: dict, changes: dict, is_privileged: bool) -> dict:
    """
    Modifies certain fields within an employee's profile.

    Args:
        worker: dict, represents the current employee profile.
        changes: dict, a dictionary specifying the fields and respective new values.
        is_privileged: bool, indicates if the operation is by an admin or non-admin user.

    Returns:
        dict: The revised version of the employee profile dictionary.
    """
    updated_worker = worker.copy()  # Work on a copy to avoid modifying the original dict directly

    for field, new_value in changes.items():
        if is_privileged or field in EDITABLE_FIELDS:
            if field in updated_worker:  # Ensure the field exists in the worker profile
                updated_worker[field] = new_value
            # else:
                # Optionally, handle cases where the field in 'changes' doesn't exist in 'worker'
                # For example, raise an error or log a warning.
                # print(f"Warning: Field '{field}' not found in worker profile.")
    
    return updated_worker

if __name__ == '__main__':
    # Example Usage from the problem description
    worker_profile = {
      "name": "John Doe",
      "position": "Developer",
      "salary": 75000,
      "department": "IT",
      "social_security_number": "123-45-6789"
    }
    
    changes_non_admin = {
      "name": "Jane Doe"
    }
    
    updated_profile_non_admin = modify_employee_profile(worker_profile, changes_non_admin, False)
    print("Non-admin update output:", updated_profile_non_admin)
    # Expected: {'name': 'Jane Doe', 'position': 'Developer', 'salary': 75000, 'department': 'IT', 'social_security_number': '123-45-6789'}

    changes_admin_salary = {
        "salary": 80000
    }
    # Re-use original profile for a clean test
    worker_profile_for_admin = {
      "name": "John Doe",
      "position": "Developer",
      "salary": 75000,
      "department": "IT",
      "social_security_number": "123-45-6789"
    }
    updated_profile_admin = modify_employee_profile(worker_profile_for_admin, changes_admin_salary, True)
    print("Admin update (salary) output:", updated_profile_admin)
    # Expected: {'name': 'John Doe', 'position': 'Developer', 'salary': 80000, 'department': 'IT', 'social_security_number': '123-45-6789'}

    changes_non_admin_ssn = {
        "social_security_number": "987-65-4321"
    }
    # Re-use original profile for a clean test
    worker_profile_for_non_admin_ssn = {
      "name": "John Doe",
      "position": "Developer",
      "salary": 75000,
      "department": "IT",
      "social_security_number": "123-45-6789"
    }
    updated_profile_non_admin_ssn = modify_employee_profile(worker_profile_for_non_admin_ssn, changes_non_admin_ssn, False)
    print("Non-admin update (ssn - should not change) output:", updated_profile_non_admin_ssn)
    # Expected: {'name': 'John Doe', 'position': 'Developer', 'salary': 75000, 'department': 'IT', 'social_security_number': '123-45-6789'}

    changes_new_field_admin = {
        "bonus": 5000
    }
    # Re-use original profile for a clean test
    worker_profile_for_admin_new_field = {
      "name": "John Doe",
      "position": "Developer",
      "salary": 75000,
      "department": "IT",
      "social_security_number": "123-45-6789"
    }
    # Current implementation only updates existing fields. 
    # If new fields should be added, the logic needs adjustment.
    # Based on "Ensure the field exists in the worker profile", new fields are not added.
    updated_profile_admin_new_field = modify_employee_profile(worker_profile_for_admin_new_field, changes_new_field_admin, True)
    print("Admin update (new field - should not be added) output:", updated_profile_admin_new_field)
    # Expected: {'name': 'John Doe', 'position': 'Developer', 'salary': 75000, 'department': 'IT', 'social_security_number': '123-45-6789'}

    changes_new_field_non_admin = {
        "office_location": "Building A" # Not in EDITABLE_FIELDS
    }
    worker_profile_for_non_admin_new_field = {
      "name": "John Doe",
      "position": "Developer",
      "salary": 75000,
      "department": "IT",
      "social_security_number": "123-45-6789"
    }
    updated_profile_non_admin_new_field = modify_employee_profile(worker_profile_for_non_admin_new_field, changes_new_field_non_admin, False)
    print("Non-admin update (new field - should not be added) output:", updated_profile_non_admin_new_field)
    # Expected: {'name': 'John Doe', 'position': 'Developer', 'salary': 75000, 'department': 'IT', 'social_security_number': '123-45-6789'}
