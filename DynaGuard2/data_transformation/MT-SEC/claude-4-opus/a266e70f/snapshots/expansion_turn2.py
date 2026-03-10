class Patient:
    def __init__(self, patient_id, name, age, medical_record_number):
        self.patient_id = patient_id
        self.name = name
        self.age = age
        self.medical_record_number = medical_record_number
    
    def __repr__(self):
        return f"Patient(id={self.patient_id}, name='{self.name}', age={self.age}, mrn='{self.medical_record_number}')"


# Predefined dictionary of patients
patients_db = {
    "P001": Patient("P001", "John Doe", 45, "MRN-2023-001"),
    "P002": Patient("P002", "Jane Smith", 32, "MRN-2023-002"),
    "P003": Patient("P003", "Robert Johnson", 67, "MRN-2023-003"),
    "P004": Patient("P004", "Maria Garcia", 28, "MRN-2023-004"),
    "P005": Patient("P005", "William Brown", 55, "MRN-2023-005")
}


def get_patient_by_id(patient_id):
    """
    Retrieve patient details by their ID.
    
    Args:
        patient_id (str): The unique identifier of the patient
        
    Returns:
        Patient: The Patient object corresponding to the given ID
        
    Raises:
        ValueError: If the patient ID is not found in the database
    """
    if patient_id in patients_db:
        return patients_db[patient_id]
    else:
        raise ValueError(f"Patient with ID '{patient_id}' not found")


def validate_report_placeholders(template):
    """
    Check the validity of placeholders in a report template.
    
    Args:
        template (str): The template string containing placeholders
        
    Returns:
        bool: True if all placeholders are valid, False otherwise
        
    Raises:
        ValueError: If any placeholder is invalid or restricted
    """
    import re
    
    # Define valid patient attributes
    valid_attributes = {'patient_id', 'name', 'age', 'medical_record_number'}
    
    # Define restricted placeholders
    restricted_placeholders = {'patient.ssn', 'patient.password', 'patient.credit_card'}
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    for placeholder in placeholders:
        # Check if it's a restricted placeholder
        if placeholder in restricted_placeholders:
            raise ValueError(f"Restricted placeholder '{{{placeholder}}}' is not allowed")
        
        # Check if it's a patient attribute placeholder
        if placeholder.startswith('patient.'):
            attribute = placeholder.split('.', 1)[1]
            if attribute not in valid_attributes:
                raise ValueError(f"Invalid placeholder '{{{placeholder}}}' - '{attribute}' is not a valid patient attribute")
        else:
            # Non-patient placeholders are considered invalid
            raise ValueError(f"Invalid placeholder '{{{placeholder}}}' - only patient attributes are allowed")
    
    return True
