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

def replace_template_placeholders(template: str, patient: Patient, allowed_attributes: set, restricted_attributes: set) -> str:
    """
    Replace placeholders in a template with patient data.
    
    Args:
        template: The template string containing placeholders
        patient: The patient object with data
        allowed_attributes: Set of attributes that can be accessed
        restricted_attributes: Set of attributes that are restricted
        
    Returns:
        The template with placeholders replaced
        
    Raises:
        ValueError: If trying to access restricted attributes
    """
    import re
    
    # Find all placeholders in the template
    placeholders = re.findall(r'\{patient\.(\w+)\}', template)
    
    # Check for any restricted data access attempts
    for placeholder in placeholders:
        if placeholder in restricted_attributes:
            raise ValueError(f"Access to restricted medical data '{placeholder}' is not allowed")
    
    # Replace allowed placeholders with actual values
    result = template
    for attr in allowed_attributes:
        placeholder = f'{{patient.{attr}}}'
        if placeholder in result:
            value = getattr(patient, attr, '')
            result = result.replace(placeholder, str(value))
    
    return result

def create_patient_report(id: str, report_template: str) -> str:
    # Check if patient ID exists
    if id not in patients:
        raise ValueError(f"Invalid patient ID: {id}")
    
    patient = patients[id]
    
    # Define allowed and restricted attributes
    allowed_attributes = {'name', 'age', 'gender', 'id'}
    restricted_attributes = {'diagnosis', 'treatment_plan'}
    
    # Replace placeholders in the template
    return replace_template_placeholders(report_template, patient, allowed_attributes, restricted_attributes)
