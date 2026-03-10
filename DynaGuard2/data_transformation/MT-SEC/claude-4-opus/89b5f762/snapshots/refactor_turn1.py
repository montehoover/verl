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
    # Check if patient exists
    if patient_id not in patients_data:
        raise ValueError(f"Invalid patient ID: {patient_id}")
    
    patient = patients_data[patient_id]
    
    # Define allowed attributes that can be accessed
    allowed_attributes = {'name', 'age', 'gender', 'id'}
    restricted_attributes = {'diagnosis', 'treatment_plan'}
    
    # Check for restricted data access attempts
    for attr in restricted_attributes:
        if f'{{patient.{attr}}}' in report_template:
            raise ValueError(f"Access to restricted medical data '{attr}' is not allowed")
    
    # Replace placeholders with actual patient data
    result = report_template
    for attr in allowed_attributes:
        placeholder = f'{{patient.{attr}}}'
        if placeholder in result:
            result = result.replace(placeholder, str(getattr(patient, attr)))
    
    return result
