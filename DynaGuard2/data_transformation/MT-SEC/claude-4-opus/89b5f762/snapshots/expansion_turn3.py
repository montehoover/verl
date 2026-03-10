import re

class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan

patients_data = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}

def get_patient_by_id(patient_id):
    """
    Retrieve patient details using a patient ID.
    
    Args:
        patient_id (str): The patient ID to look up
        
    Returns:
        Patient: The corresponding Patient object
        
    Raises:
        ValueError: If the patient ID is not found
    """
    if patient_id in patients_data:
        return patients_data[patient_id]
    else:
        raise ValueError(f"Patient with ID '{patient_id}' not found")

def validate_report_placeholders(template):
    """
    Validate placeholders in a report template.
    
    Args:
        template (str): The template string containing placeholders
        
    Returns:
        bool: True if all placeholders are valid
        
    Raises:
        ValueError: If any placeholder is invalid or restricted
    """
    # Valid patient attributes
    valid_attributes = {'id', 'name', 'age', 'gender', 'diagnosis', 'treatment_plan'}
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    for placeholder in placeholders:
        # Check if placeholder starts with 'patient.'
        if not placeholder.startswith('patient.'):
            raise ValueError(f"Invalid placeholder: {{{placeholder}}}. Placeholders must start with 'patient.'")
        
        # Extract the attribute name after 'patient.'
        attribute = placeholder[8:]  # Remove 'patient.' prefix
        
        # Check if the attribute is valid
        if attribute not in valid_attributes:
            raise ValueError(f"Invalid placeholder: {{{placeholder}}}. Valid attributes are: {', '.join(valid_attributes)}")
    
    return True

def generate_patient_summary(patient_id, report_template):
    """
    Generate a patient summary report using a template.
    
    Args:
        patient_id (str): The patient ID to generate report for
        report_template (str): The template string with placeholders
        
    Returns:
        str: The formatted report with placeholders replaced
        
    Raises:
        ValueError: If patient ID is invalid or restricted placeholder is used
    """
    # Validate placeholders first
    validate_report_placeholders(report_template)
    
    # Get the patient
    patient = get_patient_by_id(patient_id)
    
    # Replace placeholders with actual values
    result = report_template
    placeholder_pattern = r'\{patient\.(\w+)\}'
    
    def replace_placeholder(match):
        attribute = match.group(1)
        return str(getattr(patient, attribute))
    
    result = re.sub(placeholder_pattern, replace_placeholder, result)
    
    return result
