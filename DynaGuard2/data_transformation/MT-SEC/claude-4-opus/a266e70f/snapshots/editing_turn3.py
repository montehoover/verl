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
        raise ValueError(f"Invalid patient ID: {patient_id}")
    
    patient = patients[patient_id]
    
    if format_template is None:
        print(f"Patient Details for ID: {patient_id}")
        print(f"Name: {patient.name}")
        print(f"Age: {patient.age}")
        print(f"Gender: {patient.gender}")
        return
    
    # Create a dictionary of available placeholders
    placeholders = {
        'patient_id': patient_id,
        'patient_name': patient.name,
        'patient_age': patient.age,
        'patient_gender': patient.gender
    }
    
    # Replace placeholders in the template
    formatted_text = format_template
    for placeholder, value in placeholders.items():
        formatted_text = formatted_text.replace(f'{{{placeholder}}}', str(value))
    
    return formatted_text

def create_patient_report(id, report_template):
    if id not in patients:
        raise ValueError(f"Invalid patient ID: {id}")
    
    patient = patients[id]
    
    # Replace placeholders in the template
    report = report_template
    report = report.replace('{patient.id}', patient.id)
    report = report.replace('{patient.name}', patient.name)
    report = report.replace('{patient.age}', str(patient.age))
    report = report.replace('{patient.gender}', patient.gender)
    report = report.replace('{patient.diagnosis}', patient.diagnosis)
    report = report.replace('{patient.treatment_plan}', patient.treatment_plan)
    
    return report
