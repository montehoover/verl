import re

class Patient:
    def __init__(self, code, name, age, diagnosis):
        self.code = code
        self.name = name
        self.age = age
        self.diagnosis = diagnosis

# Predefined dictionary of patients
patients = {
    "P001": Patient("P001", "John Doe", 45, "Hypertension"),
    "P002": Patient("P002", "Jane Smith", 32, "Diabetes"),
    "P003": Patient("P003", "Bob Johnson", 67, "Arthritis"),
    "P004": Patient("P004", "Alice Brown", 28, "Asthma"),
    "P005": Patient("P005", "Charlie Wilson", 55, "Heart Disease")
}

def get_patient_by_code(patient_code):
    """
    Retrieves a patient object by their code.
    
    Args:
        patient_code (str): The unique patient code
        
    Returns:
        Patient: The patient object corresponding to the code
        
    Raises:
        ValueError: If the patient code does not exist
    """
    if patient_code not in patients:
        raise ValueError(f"Patient with code '{patient_code}' not found")
    
    return patients[patient_code]

def validate_report_template(template):
    """
    Validates that a report template only contains allowed placeholders.
    
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
        '{patient.diagnosis}',
        '{patient.code}'
    }
    
    # Define sensitive placeholders that should not be allowed
    sensitive_placeholders = {
        '{patient.ssn}',
        '{patient.social_security}',
        '{patient.credit_card}',
        '{patient.bank_account}',
        '{patient.password}',
        '{patient.pin}',
        '{patient.address}',
        '{patient.phone}',
        '{patient.email}'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{patient\.[a-zA-Z_]+\}'
    found_placeholders = set(re.findall(placeholder_pattern, template))
    
    # Check for sensitive placeholders
    sensitive_found = found_placeholders & sensitive_placeholders
    if sensitive_found:
        raise ValueError(f"Sensitive placeholders found: {', '.join(sensitive_found)}")
    
    # Check for invalid placeholders
    invalid_placeholders = found_placeholders - allowed_placeholders
    if invalid_placeholders:
        raise ValueError(f"Invalid placeholders found: {', '.join(invalid_placeholders)}")
    
    # Return True if all placeholders are valid
    return True
