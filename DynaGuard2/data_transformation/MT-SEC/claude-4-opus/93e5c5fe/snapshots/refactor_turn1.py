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

def compose_patient_report(patient_uid: str, report_schema: str) -> str:
    if patient_uid not in patients:
        raise ValueError(f"Patient ID {patient_uid} not found")
    
    patient = patients[patient_uid]
    
    # Check for sensitive information in the template
    if 'diagnosis' in report_schema or 'treatment_plan' in report_schema:
        raise ValueError("Attempt to access sensitive information")
    
    # Replace placeholders with patient data
    result = report_schema
    result = result.replace('{patient.id}', str(patient.id))
    result = result.replace('{patient.name}', str(patient.name))
    result = result.replace('{patient.age}', str(patient.age))
    result = result.replace('{patient.gender}', str(patient.gender))
    
    return result
