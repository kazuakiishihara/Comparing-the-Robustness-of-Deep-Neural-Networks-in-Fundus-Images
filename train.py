import prep, trainer

import numpy as np
import os
import pandas as pd

prep.set_seed(120)

# root = "./diabetic-retinopathy-detection/train001"
root = "C:/Users/Owner/Desktop/DATASET/diabetic-retinopathy-detection/train001"
# df = pd.read_csv("./diabetic-retinopathy-detection/train.csv")
df = pd.read_csv("C:/Users/Owner/Desktop/DATASET/diabetic-retinopathy-detection/train.csv")

# Predict referable DR: moderate NPDR, severe NPDR, and PDR.
df['level'] = df['level'].apply(lambda x: 0 if x in [0, 1] else 1)
df_train, df_valid, df_test = prep.split(root, df) # in-distribution data

# -----------------df_test under distribution shift (degrade image)
# root = "C:/Users/Owner/Desktop/env/2d_medical_image_DR/data/EyePACK/de_image"
# pid_to_remove = ['15337_left', '21827_left', '21827_right', '2734_right', '29640_left', '4001_left', '40764_right']
# df_test = df_test[~df_test['pid'].isin(pid_to_remove)]
# df_test['path'] = df_test['pid'].map(lambda x: os.path.join(root,'{}_001.jpeg'.format(x))) # 100: Image Blurring, 010: Retinal Artifact, 001: Light Transmission Disturbance
# ----------------

modelfiles = None
if not modelfiles:
        modelfiles = trainer.train_run(df_train, df_valid)
        print(modelfiles)
