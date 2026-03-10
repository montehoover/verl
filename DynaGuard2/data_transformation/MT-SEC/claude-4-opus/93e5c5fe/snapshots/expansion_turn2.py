import re

class Patient:
    def __init__(self, patient_id, name, age, diagnosis):
        self.patient_id = patient_id
        self.name = name
        self.age = age
        self.diagnosis = diagnosis

# Predefined dictionary of patients
patients = {
    "P001": Patient("P001", "John Doe", 45, "Hypertension"),
    "P002": Patient("P002", "Jane Smith", 32, "Diabetes"),
    "P003": Patient("P003", "Bob Johnson", 67, "Arthritis"),
    "P004": Patient("P004", "Alice Brown", 28, "Asthma"),
    "P005": Patient("P005", "Charlie Wilson", 54, "Heart Disease")
}

def get_patient_by_id(patient_id):
    """
    Retrieve a patient object by their ID.
    
    Args:
        patient_id (str): The unique identifier for the patient
        
    Returns:
        Patient: The patient object corresponding to the given ID
        
    Raises:
        ValueError: If the patient ID does not exist in the system
    """
    if patient_id not in patients:
        raise ValueError(f"Patient with ID '{patient_id}' not found")
    
    return patients[patient_id]

def validate_report_template(template):
    """
    Validate that a report template only contains allowed placeholders.
    
    Args:
        template (str): The template string to validate
        
    Returns:
        bool: True if all placeholders are valid, False otherwise
        
    Raises:
        ValueError: If any invalid or sensitive placeholders are found
    """
    # Define allowed placeholders
    allowed_placeholders = {
        '{patient.name}',
        '{patient.age}',
        '{patient.diagnosis}'
    }
    
    # Define sensitive placeholders that should not be allowed
    sensitive_placeholders = {
        '{patient.patient_id}',
        '{patient.id}',
        '{patient.ssn}',
        '{patient.social_security}',
        '{patient.address}',
        '{patient.phone}',
        '{patient.email}',
        '{patient.insurance}',
        '{patient.medical_record_number}',
        '{patient.mrn}'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{patient\.[a-zA-Z_]+\}'
    found_placeholders = set(re.findall(placeholder_pattern, template))
    
    # Check for sensitive placeholders
    sensitive_found = found_placeholders.intersection(sensitive_placeholders)
    if sensitive_found:
        raise ValueError(f"Template contains sensitive placeholders: {', '.join(sensitive_found)}")
    
    # Check for invalid placeholders
    invalid_placeholders = found_placeholders - allowed_placeholders
    if invalid_placeholders:
        raise ValueError(f"Template contains invalid placeholders: {', '.join(invalid_placeholders)}")
    
    # All placeholders are valid
    return True
