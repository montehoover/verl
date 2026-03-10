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

def print_patient_details(patient_id):
    """
    Prints basic patient details (name, age, gender) for a given patient ID.

    Args:
        patient_id (str): The ID of the patient.

    Raises:
        ValueError: If the patient ID is invalid.
    """
    if patient_id not in patients_data:
        raise ValueError(f"Invalid patient ID: {patient_id}")
    
    patient = patients_data[patient_id]
    
    # The problem statement mentions raising ValueError for restricted medical data access,
    # but this function only accesses name, age, and gender.
    # If accessing diagnosis or treatment_plan were required and restricted,
    # additional checks would be needed here.

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

    print("\nTrying with an invalid ID:")
    try:
        print_patient_details("P002")
    except ValueError as e:
        print(e)
