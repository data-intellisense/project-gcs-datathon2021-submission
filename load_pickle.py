#%% load necessary data for main.py
import pickle

# read pickle file that contains data or models
def read_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


#%% get the alias_dict, required test data
alias_dict = read_pkl(f"data/alias_dict.pickle")
las_data_TEST_renamed = read_pkl("data/las_data_TEST_renamed.pickle")
