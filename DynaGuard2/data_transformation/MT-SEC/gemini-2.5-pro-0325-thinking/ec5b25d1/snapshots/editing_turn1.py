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

def print_patient_details(patient_code):
    """
    Prints basic details (name, age, gender) for a given patient code.

    Args:
        patient_code (str): The ID of the patient.

    Raises:
        ValueError: If the patient_code is not found in the patients dictionary.
    """
    if patient_code not in patients:
        raise ValueError(f"Patient ID '{patient_code}' not found.")
    
    patient = patients[patient_code]
    
    print(f"Patient Details for Code: {patient_code}")
    print(f"  Name: {patient.name}")
    print(f"  Age: {patient.age}")
    print(f"  Gender: {patient.gender}")

if __name__ == '__main__':
    # Example usage:
    try:
        print_patient_details("P001")
    except ValueError as e:
        print(e)

    try:
        print_patient_details("P002") # This will raise a ValueError
    except ValueError as e:
        print(e)
