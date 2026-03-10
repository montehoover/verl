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

def create_person_report(patient_code: str, report_template: str) -> str:
    # Check if patient exists
    if patient_code not in patients:
        raise ValueError(f"Patient with ID {patient_code} not found")
    
    # Check for sensitive information in template
    sensitive_fields = ['diagnosis', 'treatment_plan']
    for field in sensitive_fields:
        if f'patient.{field}' in report_template:
            raise ValueError(f"Access to sensitive information '{field}' is not allowed")
    
    # Get the patient object
    patient = patients[patient_code]
    
    # Create a mapping of placeholders to values
    # Only include non-sensitive fields
    replacements = {
        '{patient.id}': patient.id,
        '{patient.name}': patient.name,
        '{patient.age}': str(patient.age),
        '{patient.gender}': patient.gender
    }
    
    # Replace all placeholders in the template
    result = report_template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    
    return result
