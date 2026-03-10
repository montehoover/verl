import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PATIENT_RECORDS = {
    "P1001": {
        "patient_id": "P1001",
        "assigned_staff": ["D101", "N201"],
        "medical_data": "Medical history for patient P1001"
    },
    "P1002": {
        "patient_id": "P1002",
        "assigned_staff": ["D102", "N202"],
        "medical_data": "Medical history for patient P1002"
    }
}


def is_authorized(staff_num: str, role: str, patient_record: dict) -> bool:
    """Check if a staff member is authorized to access a patient's record."""
    if role == 'admin':
        return True
    elif role in ['doctor', 'nurse']:
        return staff_num in patient_record['assigned_staff']
    else:
        return False


def retrieve_medical_data(patient_ref: str, patient_records: dict) -> dict:
    """Retrieve patient record if it exists."""
    if patient_ref not in patient_records:
        return None
    return patient_records[patient_ref]


def get_patient_data(staff_num: str, role: str, patient_ref: str):
    # Log access attempt
    logger.info(f"Access attempt - Staff: {staff_num}, Role: {role}, Patient: {patient_ref}")
    
    # Retrieve patient record
    patient_record = retrieve_medical_data(patient_ref, PATIENT_RECORDS)
    
    if patient_record is None:
        logger.warning(f"Access denied - Staff: {staff_num}, Role: {role}, Patient: {patient_ref} - Reason: Patient not found")
        raise PermissionError("Unauthorized access")
    
    # Check authorization
    if is_authorized(staff_num, role, patient_record):
        logger.info(f"Access granted - Staff: {staff_num}, Role: {role}, Patient: {patient_ref}")
        return patient_record['medical_data']
    else:
        logger.warning(f"Access denied - Staff: {staff_num}, Role: {role}, Patient: {patient_ref} - Reason: Insufficient permissions")
        raise PermissionError("Unauthorized access")
