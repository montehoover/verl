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
    if patient_code not in patients:
        raise ValueError(f"Patient with ID {patient_code} not found")
    
    patient = patients[patient_code]
    
    # Check if template contains sensitive fields
    if 'diagnosis' in report_template or 'treatment_plan' in report_template:
        raise ValueError("Access to sensitive information (diagnosis or treatment_plan) is not allowed")
    
    # Replace placeholders with patient data
    result = report_template
    result = result.replace('{patient.id}', str(patient.id))
    result = result.replace('{patient.name}', str(patient.name))
    result = result.replace('{patient.age}', str(patient.age))
    result = result.replace('{patient.gender}', str(patient.gender))
    
    return result
