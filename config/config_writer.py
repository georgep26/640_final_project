import json
import os

class ConfigWriter:
    def __init__(self, output_path):
        self.output_path = output_path
        self.config = {}
        self.out_string = ""

    def add(self, title, item):
        if title in self.config.keys():
            self.config[title] = self.config[title].extend(item)
        else:
            self.config[title] = item

    def print(self, print_string):
        self.out_string = self.out_string + "\n" + print_string
        self.write()

    def write(self):
        with open(os.path.join(self.output_path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)
        with open(os.path.join(self.output_path, "output.txt"), "w") as f:
            f.write(self.out_string)