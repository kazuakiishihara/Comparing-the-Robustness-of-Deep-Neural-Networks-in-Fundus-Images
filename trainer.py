import prep

import numpy as np
from sklearn.metrics import roc_auc_score
import time
import timm
import timm.scheduler
import torch
from torch import nn
from torch.utils import data as torch_data
from torchvision import transforms
from transformers import ResNetForImageClassification, SwinForImageClassification

class Dataset(torch_data.Dataset):
    def __init__(self, csv, transform=None):
        self.csv = csv
        self.transform = transform
    
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        img_path = self.csv["path"].values[idx]
        img = prep.load_ben_color(img_path)
        if self.transform:
            img = transforms.ToPILImage()(img)
            img = self.transform(img)
        y = self.csv["level"].values[idx]
        return {"X" : img, "y" : torch.tensor(y).long()}
    
class Trainer:
    def __init__(
        self, 
        model, 
        device, 
        optimizer, 
        criterion,
        scheduler
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.best_valid_score = np.inf
        self.n_patience = 0
        self.lastmodel = None
        
    def fit(self, epochs, train_loader, valid_loader, patience):

        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)
            self.scheduler.step(n_epoch)
            
            train_loss, train_time, train_auc = self.train_epoch(train_loader)
            valid_loss, valid_time, valid_auc = self.valid_epoch(valid_loader)
            
            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, time: {:.2f} s, AUC: {:.4f}",
                n_epoch, train_loss, train_time, train_auc
            )
            
            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, time: {:.2f} s, AUC: {:.4f}",
                n_epoch, valid_loss, valid_time, valid_auc
            )

            if self.best_valid_score > valid_loss: 
                self.save_model(n_epoch)
                self.info_message(
                    "auc improved from {:.4f} to {:.4f}. Saved model to '{}'", 
                    self.best_valid_score, valid_loss, self.lastmodel
                )
                self.best_valid_score = valid_loss
                self.n_patience = 0
            else:
                self.n_patience += 1
            
            if self.n_patience >= patience:
                self.info_message("\nValid auc didn't improve last {} epochs.", patience)
                break
            
    def train_epoch(self, train_loader):
        self.model.train()
        scaler = torch.cuda.amp.GradScaler()
        t = time.time()
        sum_loss = 0

        prob_li = []
        labels_li = []

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device, non_blocking=True)
            targets = batch["y"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.model(X).logits.squeeze(1) # if transformers model
                # outputs = self.model(X).squeeze(1) #if timm model

                prob = torch.sigmoid(outputs)
                prob_li.extend(prob.detach().clone().to('cpu').tolist())
                labels_li.extend(targets.detach().clone().to('cpu').tolist())

                loss = self.criterion(outputs, targets.float())
                sum_loss += loss.detach().item()

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            message = 'Train Step {}/{}, train_loss: {:.4f}'
            self.info_message(message, step, len(train_loader), sum_loss/step, end="\r")
        
        return sum_loss/len(train_loader), int(time.time() - t), roc_auc_score(labels_li, prob_li)
    
    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        sum_loss = 0

        prob_li = []
        labels_li = []

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device, non_blocking=True)
                targets = batch["y"].to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = self.model(X).logits.squeeze(1) # if transformers model
                    # outputs = self.model(X).squeeze(1) #if timm model

                    prob = torch.sigmoid(outputs)
                    prob_li.extend(prob.detach().clone().to('cpu').tolist())
                    labels_li.extend(targets.detach().clone().to('cpu').tolist())

                    loss = self.criterion(outputs, targets.float())
                    sum_loss += loss.detach().item()

            message = 'Valid Step {}/{}, valid_loss: {:.4f}'
            self.info_message(message, step, len(valid_loader), sum_loss/step, end="\r")
        
        return sum_loss/len(valid_loader), int(time.time() - t), roc_auc_score(labels_li, prob_li)
    
    def save_model(self, n_epoch):
        self.lastmodel = "./trained_model/resnet-152.pth" # name of a trained model
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            self.lastmodel,
        )

    def info_message(self, message, *args, end="\n"):
        print(message.format(*args), end=end)

def train_run(df_train, df_valid):
    # torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-120, 120)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset = Dataset(df_train, train_transform)
    valid_dataset = Dataset(df_valid, test_transform)

    train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    
    # ResNet: [resnet-50, resnet-101, resnet-152]
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-152", num_labels=1, ignore_mismatched_sizes=True)

    # Swin-Transformer: [swin-tiny-patch4-window7-224, swin-small-patch4-window7-224, swin-base-patch4-window7-224, swin-large-patch4-window7-224]
    # model = SwinForImageClassification.from_pretrained("microsoft/swin-small-patch4-window7-224", num_labels=1, ignore_mismatched_sizes=True)

    # MLP-Mixer: [mixer_b16_224.goog_in21k, mixer_l16_224.goog_in21k]
    # model = timm.create_model("mixer_l16_224.goog_in21k", pretrained=True, num_classes=1)
            
    model.to(device, non_blocking=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = timm.scheduler.StepLRScheduler(
                                                optimizer, 
                                                decay_t=30, 
                                                decay_rate=0.1, 
                                                warmup_lr_init=1e-4, 
                                                warmup_t=20)
    
    trainer = Trainer(model, device, optimizer, criterion, scheduler)
    history = trainer.fit(100, train_loader, valid_loader, 100)
    return trainer.lastmodel
