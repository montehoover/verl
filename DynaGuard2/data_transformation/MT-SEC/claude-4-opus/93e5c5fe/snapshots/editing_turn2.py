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

def print_patient_details(patient_id, format_template=None):
    if patient_id not in patients:
        raise ValueError(f"Patient ID '{patient_id}' not found")
    
    patient = patients[patient_id]
    
    if format_template is None:
        print(f"Patient Details for ID: {patient_id}")
        print(f"Name: {patient.name}")
        print(f"Age: {patient.age}")
        print(f"Gender: {patient.gender}")
        return
    
    # Create a dictionary with all available placeholders
    replacements = {
        'patient_id': patient_id,
        'patient_name': patient.name,
        'patient_age': str(patient.age),
        'patient_gender': patient.gender
    }
    
    # Replace placeholders in the template
    formatted_string = format_template
    for placeholder, value in replacements.items():
        formatted_string = formatted_string.replace(f'{{{placeholder}}}', value)
    
    print(formatted_string)
    return formatted_string
