import pickle

def process_serialfile(input_path, raw_export=False):
    with open(input_path, 'rb') as f:
        data = f.read()
    
    if raw_export:
        return data
    
    try:
        return pickle.loads(data)
    except:
        return data
