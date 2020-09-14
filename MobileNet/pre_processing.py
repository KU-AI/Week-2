import os
from pathlib import Path
import shutil

class config():
    def __init__(self):
        self.data_path = "./data/pre_data/"
        self.img_path = "./data/weather/dataset/"
        Path(self.data_path+"train").mkdir(exist_ok=True)
        Path(self.data_path+"train").mkdir(exist_ok=True)

if __name__ == "__main__":
    cfg = config()
    class_list = os.listdir(cfg.img_path)

    for idx, i in enumerate(class_list):
        for j in os.listdir(cfg.img_path+i):
            idx += 1
            Path(cfg.data_path+"train/"+i).mkdir(parents=True, exist_ok=True)
            Path(cfg.data_path+"val/"+i).mkdir(parents=True, exist_ok=True)
            if (idx%2) == 0:
                shutil.copyfile(cfg.img_path+i+'/'+j, cfg.data_path+"train/"+i+'/'+j)
            elif (idx%2) == 1:
                shutil.copyfile(cfg.img_path+i+'/'+j, cfg.data_path+"val/"+i+'/'+j)
