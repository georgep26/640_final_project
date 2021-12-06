import os
import config.constants as cst
import pandas as pd

class MasterLog():

    def __init__(self):
        self.log_file_path = os.path.join(cst.output_dir, "master_log.txt")
        
        if os.path.exists(self.log_file_path):
            self.log = pd.read_csv(self.log_file_path, index_col=1)
        else:
            self.log = pd.DataFrame()

    def add_dict(self, dict):
        new_df = pd.DataFrame(dict, index=[0])
        self.log = self.log.append(new_df).reset_index(drop=True)

    def write_log(self):
        self.log.to_csv(self.log_file_path)
