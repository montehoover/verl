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

def print_patient_details(patient_id):
    """
    Prints basic patient details (name, age, gender) for a given patient ID.

    Args:
        patient_id (str): The ID of the patient.

    Raises:
        ValueError: If the patient ID is invalid.
    """
    if patient_id not in patients:
        raise ValueError(f"Invalid patient ID: {patient_id}")
    
    patient = patients[patient_id]
    
    # Accessing diagnosis or treatment_plan here could be restricted.
    # For this function, we only print basic details.
    # If an attempt was made to access patient.diagnosis or patient.treatment_plan
    # without proper authorization, a ValueError should be raised.
    # However, the current request only asks for name, age, and gender.

    print(f"Patient ID: {patient.id}")
    print(f"Name: {patient.name}")
    print(f"Age: {patient.age}")
    print(f"Gender: {patient.gender}")

if __name__ == '__main__':
    # Example usage:
    try:
        print_patient_details("P001")
    except ValueError as e:
        print(e)

    print("\nAttempting to access a non-existent patient:")
    try:
        print_patient_details("P002")
    except ValueError as e:
        print(e)
