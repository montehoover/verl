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

def generate_patient_summary(patient_id, report_template):
    if patient_id not in patients_data:
        raise ValueError(f"Invalid patient ID: {patient_id}")
    
    patient = patients_data[patient_id]
    
    # Replace placeholders in the template
    formatted_report = report_template
    formatted_report = formatted_report.replace('{patient.id}', str(patient.id))
    formatted_report = formatted_report.replace('{patient.name}', str(patient.name))
    formatted_report = formatted_report.replace('{patient.age}', str(patient.age))
    formatted_report = formatted_report.replace('{patient.gender}', str(patient.gender))
    formatted_report = formatted_report.replace('{patient.diagnosis}', str(patient.diagnosis))
    formatted_report = formatted_report.replace('{patient.treatment_plan}', str(patient.treatment_plan))
    
    return formatted_report
