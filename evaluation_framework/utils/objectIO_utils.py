import pickle


def save_obj(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
