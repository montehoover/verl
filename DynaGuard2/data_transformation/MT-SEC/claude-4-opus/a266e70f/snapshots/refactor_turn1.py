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

def create_patient_report(id: str, report_template: str) -> str:
    # Check if patient ID exists
    if id not in patients:
        raise ValueError(f"Invalid patient ID: {id}")
    
    patient = patients[id]
    
    # Define allowed attributes that can be accessed in templates
    allowed_attributes = {'name', 'age', 'gender', 'id'}
    restricted_attributes = {'diagnosis', 'treatment_plan'}
    
    # Create a copy of the template to work with
    result = report_template
    
    # Find all placeholders in the template
    import re
    placeholders = re.findall(r'\{patient\.(\w+)\}', report_template)
    
    # Check for any restricted data access attempts
    for placeholder in placeholders:
        if placeholder in restricted_attributes:
            raise ValueError(f"Access to restricted medical data '{placeholder}' is not allowed")
    
    # Replace allowed placeholders with actual values
    for attr in allowed_attributes:
        placeholder = f'{{patient.{attr}}}'
        if placeholder in result:
            value = getattr(patient, attr, '')
            result = result.replace(placeholder, str(value))
    
    return result
