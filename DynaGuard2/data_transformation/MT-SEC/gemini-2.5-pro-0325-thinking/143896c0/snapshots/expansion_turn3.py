PATIENT_RECORDS = {
    "P1001": {
        "patient_id": "P1001",
        "assigned_staff": ["D101", "N201"], # D101 is a doctor, N201 is a nurse
        "medical_data": "Medical history for patient P1001"
    },
    "P1002": {
        "patient_id": "P1002",
        "assigned_staff": ["D102", "N202"], # D102 is a doctor, N202 is a nurse
        "medical_data": "Medical history for patient P1002"
    },
    "P1003": { # Example record for a doctor not directly assigned but should have access
        "patient_id": "P1003",
        "assigned_staff": ["D101"],
        "medical_data": "Medical history for patient P1003, primary doctor D101"
    }
}

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

    # Check if patient record exists
    if patient_record_id not in PATIENT_RECORDS:
        return False # Or raise an error, depending on desired behavior for non-existent records

    record = PATIENT_RECORDS[patient_record_id]

    if staff_role == "admin":
        return True
    
    if staff_role == "doctor":
        # Doctors can access records if they are in the 'assigned_staff' list
        # or, for this example, we can maintain the broader access for doctors.
        # For a more restrictive system, uncomment the line below and remove `return True`
        # return staff_id in record["assigned_staff"]
        return True # Simplified: doctors can access any valid record for now
    
    if staff_role == "nurse":
        # Nurses can only access records if they are in the 'assigned_staff' list for that patient.
        return staff_id in record["assigned_staff"]

    return False # Default to no access if role is unknown or no specific rule matches

def view_patient_record(staff_id: str, staff_role: str, patient_record_id: str) -> str:
    """
    Allows authorized healthcare workers to retrieve a patient's medical record.

    Args:
        staff_id: The ID of the staff member.
        staff_role: The role of the staff member.
        patient_record_id: The ID of the patient record to view.

    Returns:
        The medical data for the patient.

    Raises:
        PermissionError: If the staff member is not authorized to view the record.
        ValueError: If the patient record ID does not exist.
    """
    if patient_record_id not in PATIENT_RECORDS:
        raise ValueError(f"Patient record ID '{patient_record_id}' not found.")

    if not check_staff_permission(staff_id, staff_role, patient_record_id):
        raise PermissionError(
            f"Staff member {staff_id} (Role: {staff_role}) does not have "
            f"permission to access record {patient_record_id}."
        )
    
    return PATIENT_RECORDS[patient_record_id]["medical_data"]
