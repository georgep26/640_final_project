import os
import config.constants as cst
import pandas as pd

class MasterLog():

    def __init__(self):
        self.log_file_path = os.path.join(cst.output_dir, "master_log.txt")
        
        if os.path.exists(self.log_file_path):
            self.log = pd.read_csv(self.log_file_path)
        else:
            self.log = pd.DataFrame()

    def add_dict(self, dict):
        new_df = pd.DataFrame.from_dict(dict, index=[0])
        self.log = self.log.append(new_df)

    def write_log(self):
        self.log.to_csv(self.log_file_path)
