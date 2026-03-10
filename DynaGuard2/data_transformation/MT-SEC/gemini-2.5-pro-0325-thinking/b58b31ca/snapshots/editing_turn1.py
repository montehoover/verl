# Assume PATIENT_RECORDS is available in the setup.
# Example structure:
# PATIENT_RECORDS = {
#     "record_id_1": {"name": "Patient Alpha", "assigned_staff": ["worker_id_A", "worker_id_B"]},
#     "record_id_2": {"name": "Patient Beta", "assigned_staff": ["worker_id_C"]},
# }
PATIENT_RECORDS = {} # Placeholder, to be populated or defined elsewhere

def is_assigned_to_patient(worker_id: str, patient_record_id: str) -> bool:
    """
    Checks if a healthcare worker is assigned to a specific patient.

    Args:
        worker_id: The ID of the healthcare worker.
        patient_record_id: The ID of the patient record.

    Returns:
        True if the worker is listed in the patient's assigned staff, otherwise False.
    """
    patient_record = PATIENT_RECORDS.get(patient_record_id)
    if patient_record:
        assigned_staff = patient_record.get("assigned_staff", [])
        return worker_id in assigned_staff
    return False
