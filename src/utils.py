from sklearn.metrics import r2_score
import pickle
import os

def save_file_as_pickle(path,name):
    dir_name=os.path.dirname(path)
    os.makedirs(dir_name,exist_ok=True)
    with open(path,"wb") as path_obj:
        pickle.dump(name,path_obj)



def load_pickle_file(path):
    with open(path,"rb") as path_obj:
        return pickle.load(path_obj)

def evaluate_model(true_value,predicted_value):
    r2=r2_score(true_value,predicted_value)
    return r2