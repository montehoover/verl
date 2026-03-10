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

def is_staff_authorized(staff_id: str, staff_role: str, patient_record: dict) -> bool:
    """Check if a staff member is authorized to access a patient's record."""
    if staff_role == 'admin':
        return True
    
    if staff_role in ['doctor', 'nurse']:
        return staff_id in patient_record['assigned_staff']
    
    return False

def get_patient_record(patient_id: str) -> dict:
    """Retrieve a patient record by ID."""
    if patient_id not in PATIENT_RECORDS:
        raise PermissionError("Patient record not found")
    
    return PATIENT_RECORDS[patient_id]

def retrieve_patient_data(staff_id: str, staff_role: str, patient_id: str):
    logger.info(f"Access attempt: staff_id={staff_id}, staff_role={staff_role}, patient_id={patient_id}")
    
    try:
        patient_record = get_patient_record(patient_id)
    except PermissionError as e:
        logger.warning(f"Access denied: staff_id={staff_id}, staff_role={staff_role}, patient_id={patient_id}, reason=Patient record not found")
        raise
    
    if is_staff_authorized(staff_id, staff_role, patient_record):
        logger.info(f"Access granted: staff_id={staff_id}, staff_role={staff_role}, patient_id={patient_id}")
        return patient_record['medical_data']
    
    if staff_role in ['doctor', 'nurse']:
        logger.warning(f"Access denied: staff_id={staff_id}, staff_role={staff_role}, patient_id={patient_id}, reason=Staff not assigned to patient")
        raise PermissionError("Unauthorized access: staff not assigned to this patient")
    else:
        logger.warning(f"Access denied: staff_id={staff_id}, staff_role={staff_role}, patient_id={patient_id}, reason=Role not permitted")
        raise PermissionError("Unauthorized access: role not permitted to view medical data")
