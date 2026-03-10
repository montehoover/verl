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

def generate_patient_summary(patient_id: str, report_template: str) -> str:
    """
    Generate a patient summary by filling in a template with patient information.
    
    Args:
        patient_id: The unique identifier of the patient within the system.
        report_template: The string template that will be filled with patient information.
    
    Returns:
        A string with the patient's information filled into the template.
    
    Raises:
        ValueError: If the patient ID is invalid or if an attempt is made to access restricted medical data.
    """
    # Check if patient exists
    if patient_id not in patients_data:
        raise ValueError(f"Invalid patient ID: {patient_id}")
    
    patient = patients_data[patient_id]
    
    # Check for restricted fields in the template
    restricted_fields = ['diagnosis', 'treatment_plan']
    for field in restricted_fields:
        if f'{{patient.{field}}}' in report_template:
            raise ValueError(f"Access to restricted medical data '{field}' is not allowed")
    
    # Create a dictionary with allowed patient attributes
    allowed_data = {
        'patient.id': patient.id,
        'patient.name': patient.name,
        'patient.age': patient.age,
        'patient.gender': patient.gender
    }
    
    # Replace placeholders in the template
    result = report_template
    for key, value in allowed_data.items():
        placeholder = f'{{{key}}}'
        result = result.replace(placeholder, str(value))
    
    return result
