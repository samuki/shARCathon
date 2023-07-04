import os
import json
import shutil

def filter_files(src_directory, dest_directory, n, greater=False):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
        
    for filename in os.listdir(src_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(src_directory, filename)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            if greater:
                if 'test' in data and len(data['test'][0]['input'])*len(data['test'][0]['input'][0]) > n:
                    shutil.copy(file_path, dest_directory)
            else:
                if 'test' in data and len(data['test'][0]['input'])*len(data['test'][0]['input'][0]) <= n:
                    shutil.copy(file_path, dest_directory)

n_small = 50  
n_medium = 100  

filter_files('data/training/', 'data/training_small/', n_small)
filter_files('data/evaluation/', 'data/evaluation_small/', n_small)

filter_files('data/training/', 'data/training_medium/', n_medium)
filter_files('data/evaluation/', 'data/evaluation_medium/', n_medium)

filter_files('data/training/', 'data/training_large/', n_medium, greater=True)
filter_files('data/evaluation/', 'data/evaluation_large/', n_medium, greater=True)
