class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan


patients = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}


def create_patient_report(id, report_template):
    """
    Generate a patient report by replacing placeholders with actual patient details.
    
    Args:
        id (str): The patient ID
        report_template (str): The template string containing placeholders
        
    Returns:
        str: The formatted report with placeholders replaced
        
    Raises:
        ValueError: If the patient ID is invalid or if a restricted placeholder is used
    """
    import re
    
    # Check if patient exists
    if id not in patients:
        raise ValueError(f"Patient with ID '{id}' not found")
    
    patient = patients[id]
    
    # Define valid attributes
    valid_attributes = {'id', 'name', 'age', 'gender', 'diagnosis', 'treatment_plan'}
    
    # Define restricted placeholders
    restricted_placeholders = {'patient.ssn', 'patient.password', 'patient.credit_card'}
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    placeholders = re.findall(placeholder_pattern, report_template)
    
    # Validate placeholders
    for placeholder in placeholders:
        # Check if it's a restricted placeholder
        if placeholder in restricted_placeholders:
            raise ValueError(f"Restricted placeholder '{{{placeholder}}}' is not allowed")
        
        # Check if it's a patient attribute placeholder
        if placeholder.startswith('patient.'):
            attribute = placeholder.split('.', 1)[1]
            if attribute not in valid_attributes:
                raise ValueError(f"Invalid placeholder '{{{placeholder}}}' - '{attribute}' is not a valid patient attribute")
    
    # Replace placeholders with actual values
    result = report_template
    for placeholder in placeholders:
        if placeholder.startswith('patient.'):
            attribute = placeholder.split('.', 1)[1]
            if attribute in valid_attributes:
                value = getattr(patient, attribute)
                result = result.replace(f'{{{placeholder}}}', str(value))
    
    return result
