import numpy as np
import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from Config import DATA_PATH, actions, no_sequences

if os.path.exists(DATA_PATH):
    # Delete the folder and all its contents
    shutil.rmtree(DATA_PATH)


for action in actions: 
    action_path = os.path.join(DATA_PATH, action)
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(action_path):
        os.makedirs(action_path)
        dirmax = 0
    else:
        dirmax = np.max(np.array(os.listdir(action_path)).astype(int))
    
    for sequence in range(1, no_sequences + 1):
        try: 
            os.makedirs(os.path.join(action_path, str(dirmax + sequence)))
        except FileExistsError:
            pass
