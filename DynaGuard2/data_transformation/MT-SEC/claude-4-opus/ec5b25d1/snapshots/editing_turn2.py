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

def print_patient_details(patient_code, format_template=None):
    if patient_code not in patients:
        raise ValueError(f"Patient ID '{patient_code}' not found")
    
    patient = patients[patient_code]
    
    if format_template is None:
        print(f"Patient Details for {patient_code}")
        print(f"Name: {patient.name}")
        print(f"Age: {patient.age}")
        print(f"Gender: {patient.gender}")
        return
    
    # Create placeholder mappings
    placeholders = {
        '{patient_name}': patient.name,
        '{patient_age}': str(patient.age),
        '{patient_gender}': patient.gender,
        '{patient_id}': patient.id
    }
    
    # Replace placeholders in template
    result = format_template
    for placeholder, value in placeholders.items():
        result = result.replace(placeholder, value)
    
    return result
