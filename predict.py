import calibration_metrics as caltrain
import trainer

import numpy as np
# import os
# import pandas as pd
from sklearn.metrics import roc_auc_score
import timm
import time
import torch
from torchvision import transforms
from transformers import ResNetForImageClassification, SwinForImageClassification

def predict_logits(df, modelfiles, model_type):
    print("Predict:", df.shape)
    t = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    data_dataset = trainer.Dataset(df, test_transform)

    data_loader = torch.utils.data.DataLoader(data_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    if model_type == "resnet":
        model = ResNetForImageClassification.from_pretrained("microsoft/"+modelfiles, num_labels=1, ignore_mismatched_sizes=True)
    elif model_type == "swin":
        model = SwinForImageClassification.from_pretrained("microsoft/"+modelfiles, num_labels=1, ignore_mismatched_sizes=True)
    else:
        model = timm.create_model(modelfiles, num_classes=1)
    model.to(device, non_blocking=True)
    
    checkpoint = torch.load("./trained_model/"+modelfiles + ".pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    y_pred = []
    ids = []

    for e, batch in enumerate(data_loader,1):
        print(f"{e}/{len(data_loader)}", end="\r")
        with torch.no_grad():
            if (model_type=="resnet") or (model_type=="swin"):
                tmp_pred = model(batch["X"].to(device, non_blocking=True)).logits.squeeze(1).detach().clone().to('cpu')
            else:
                tmp_pred = model(batch["X"].to(device, non_blocking=True)).squeeze(1).detach().clone().to('cpu')
            y_pred.extend(tmp_pred.tolist())
            ids.extend(batch["y"].detach().clone().to('cpu').tolist())
            
    return torch.tensor(y_pred), torch.tensor(ids), int(time.time() - t) # logits, label, time

def evaluate(logits, label):
    prob, label = torch.sigmoid(logits).numpy(), label.numpy()
    prob_ = prob[:, np.newaxis]
    label_ = label[:, np.newaxis]

    em_ece_sweep_L1 = caltrain.CalibrationMetric(
        ce_type="em_ece_sweep", bin_method="equal_examples", norm=1, multiclass_setting="marginal")
    ece_sweep = em_ece_sweep_L1.compute_error(prob_, label_)
    roc = roc_auc_score(label, prob)
    return ece_sweep, roc
