import logging
from datetime import datetime

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

# Configure logging
logging.basicConfig(
    filename='patient_access.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def is_authorized(worker_id: str, worker_role: str, patient_record: dict) -> bool:
    """Check if a worker is authorized to access a patient record."""
    # Admins can access all records
    if worker_role.lower() == 'admin':
        return True
    
    # Other roles need to be assigned to the patient
    return worker_id in patient_record.get("assigned_staff", [])

def retrieve_medical_data(patient_record: dict) -> str:
    """Extract medical data from a patient record."""
    return patient_record["medical_data"]

def get_patient_data(worker_id: str, worker_role: str, record_id: str):
    # Log the access attempt
    logging.info(f"Access attempt - Worker ID: {worker_id}, Role: {worker_role}, Record ID: {record_id}")
    
    # Check if the record exists
    if record_id not in PATIENT_RECORDS:
        logging.warning(f"Access denied - Worker ID: {worker_id}, Role: {worker_role}, Record ID: {record_id}, Reason: Patient record not found")
        raise PermissionError("Patient record not found")
    
    patient_record = PATIENT_RECORDS[record_id]
    
    # Check if worker has permission to access the record
    if not is_authorized(worker_id, worker_role, patient_record):
        logging.warning(f"Access denied - Worker ID: {worker_id}, Role: {worker_role}, Record ID: {record_id}, Reason: Insufficient permissions")
        raise PermissionError("Insufficient permissions to access this patient record")
    
    # Log successful access
    logging.info(f"Access granted - Worker ID: {worker_id}, Role: {worker_role}, Record ID: {record_id}")
    
    # Return the medical data
    return retrieve_medical_data(patient_record)
