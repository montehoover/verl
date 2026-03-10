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

def create_person_report(patient_code, report_template):
    if patient_code not in patients:
        raise ValueError(f"Patient ID '{patient_code}' not found")
    
    patient = patients[patient_code]
    
    # Replace placeholders with actual patient data
    result = report_template
    result = result.replace('{patient.name}', patient.name)
    result = result.replace('{patient.age}', str(patient.age))
    result = result.replace('{patient.gender}', patient.gender)
    result = result.replace('{patient.id}', patient.id)
    result = result.replace('{patient.diagnosis}', patient.diagnosis)
    result = result.replace('{patient.treatment_plan}', patient.treatment_plan)
    
    return result
