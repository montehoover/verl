def authenticate_user(staff_id: str, password: str) -> bool:
    """
    Authenticates a user based on staff_id and password.

    Args:
        staff_id: The staff ID of the user.
        password: The password of the user.

    Returns:
        True if the credentials are correct, False otherwise.
    """
    # In a real application, you would look up the staff_id
    # in a database and compare the hashed password.
    # For this example, we'll use a hardcoded credential.
    if staff_id == "admin" and password == "password123":
        return True
    return False

def check_staff_permission(staff_id: str, staff_role: str, patient_record_id: str) -> bool:
    """
    Checks if a staff member has permission to access a patient's record.

    Args:
        staff_id: The ID of the staff member.
        staff_role: The role of the staff member (e.g., 'doctor', 'nurse', 'admin').
        patient_record_id: The ID of the patient record.

    Returns:
        True if the staff member has permission, False otherwise.
    """
    # In a real application, this would involve more complex logic:
    # 1. Verify the staff_id is valid.
    # 2. Look up the staff_role associated with the staff_id.
    # 3. Check if the staff_role has general permission to view patient records.
    # 4. Check if this specific staff member is assigned to this patient (e.g., primary care physician).
    # 5. Admins might have blanket access.

    # For this example, let's implement some basic rules:
    # - Admins can access any record.
    # - Doctors can access any record (simplified for now).
    # - Nurses can only access records if they are 'assigned' to the patient.
    #   (We'll simulate this by checking if the patient_record_id contains the nurse's staff_id,
    #    which is a placeholder for a real assignment check).

    if staff_role == "admin":
        return True
    
    if staff_role == "doctor":
        # In a real system, doctors might only access records of patients under their care.
        # For this example, we'll allow doctors to access all records.
        return True
    
    if staff_role == "nurse":
        # Simulate checking if the nurse is assigned to the patient.
        # This is a placeholder for a more robust check (e.g., querying a database table
        # that links nurses to patients).
        # Let's assume a patient_record_id like "PAT123_NURSE_staff_id_XYZ" indicates assignment.
        if f"NURSE_{staff_id}" in patient_record_id:
            return True
        # Or, a simpler rule: nurse can access if their ID is part of the record ID (very basic)
        if staff_id in patient_record_id: # Simplified for example
            return True 
        return False

    return False # Default to no access if role is unknown or no specific rule matches
