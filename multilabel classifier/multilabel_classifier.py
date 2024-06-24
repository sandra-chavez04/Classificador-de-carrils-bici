import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import wandb
import os
import random
import shutil
from torchvision.models import resnet50
from torch.utils.data import TensorDataset

if not torch.cuda.is_available():
    raise Exception("You should enable GPU in the Runtime menu")
device = torch.device("cuda")

api_key =  "your_wandb api key"
wandb.login(key=api_key)

############################################################################################################
                                        ## Define the model ##
############################################################################################################

# Define the model
class MultiLabelModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize the model with the number of labels
num_classes = 6  
model = MultiLabelModel(num_classes)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()

# Define the class
class MultiLabelDataset(Dataset):
    def __init__(self, labels_file_path, dataset_path, transform=None):
        self.data = []
        self.transform = transform
        
        with open(labels_file_path, 'r') as file:
            for line in file:
                line_data = line.strip().split()
                image_path = os.path.join(dataset_path,line_data[0])
                labels = [int(label) for label in line_data[-6:]]  # Get the last six columns as labels
                self.data.append((image_path, labels))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, labels = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels

transform = transforms.Compose([
                                transforms.Resize(380), # Resize the short side of the image to 380 keeping aspect ratio
                                transforms.CenterCrop(380), # Crop a square in the center of the image
                                transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
                                ])


############################################################################################################
                                    ## Image Organization ##
############################################################################################################

labels_file_path = './labels.txt'
dataset_path = 'path where /LaneClassifier/multilabel_classifier is located'

# Create custom data set
dataset = MultiLabelDataset(labels_file_path, dataset_path, transform=transform)

# Calculate the size of the train and validation sets
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = dataset_size - train_size

# Create indexes to split the data
indexes = list(range(dataset_size))
np.random.shuffle(indexes)
train_indexes = indexes[:train_size]
val_indexes = indexes[train_size:]

# Create subsets for train and validation and create DataLoader for training set and validation set
batch_size = 64

train_set = torch.utils.data.Subset(dataset, train_indexes)
val_set = torch.utils.data.Subset(dataset, val_indexes)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Check the size of the training set and validation set
print('Amount of data in training set:', len(train_set))
print('Amount of data in validation set:', len(val_set))

# Check an example of the training set and validation set
for images, labels in train_loader:
    print('Training batch size:', images.shape)
    print('Training batch labels:', labels.shape)
    break
for images, labels in val_loader:
    print('Validation batch size:', images.shape)
    print('Validation batch labels:', labels.shape)
    break


############################################################################################################
                                        ## Model training ##
############################################################################################################

pretrained_model = resnet50(pretrained=True)
pretrained_model.eval()
pretrained_model.to(device)

# Take the first part of the model to use it as a feature_extractor
feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-1])

# Freeze some layers of the base model
for param in feature_extractor.parameters():
    param.requires_grad = False
for param in feature_extractor[-3:].parameters():
    param.requires_grad = True

# Add a custom classifier
num_ftrs = pretrained_model.fc.in_features
feature_classifier = nn.Sequential(
                        nn.Linear(2048, 512),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(512, num_classes)
                        #nn.Sigmoid()
)

model = nn.Sequential(
            feature_extractor,
            nn.Flatten(),
            feature_classifier
)

model.to(device)


############################# Required memory ####################################

total_params = 0
for p in model.parameters():
    total_params += p.numel()

print("Total number of model parameters:", total_params)
#################################################################################

########################### Weights of each class ################################
with open('./text2multilabel.txt', 'r') as f:
    lines = f.readlines()

count_column1 = 0
count_column2 = 0
count_column3 = 0
count_column4 = 0
count_column5 = 0
count_column6 = 0

for line in lines:
    # Split the line into columns
    columns = line.strip().split()
    # Add the ones in the last 6 columns
    count_column1 += int(columns[-6])
    count_column2 += int(columns[-5])
    count_column3 += int(columns[-4])
    count_column4 += int(columns[-3])
    count_column5 += int(columns[-2])
    count_column6 += int(columns[-1])

# Calculate class weights
total_samples = len(dataset)
class_counts = [count_column1, count_column2, count_column3, count_column4, count_column5, count_column6]

class_weights = []
for count in class_counts:
    weight = total_samples / (2 * count)
    class_weights.append(weight)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)

#################################################################################

# Now we proceed with fine tuning:

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = 64
train_dataset = torch.utils.data.Subset(dataset, train_indexes)
val_dataset = torch.utils.data.Subset(dataset, val_indexes)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


optimizer = optim.Adam(model.parameters(), lr=1e-4)


############################ Inference time ####################################

import time

def inference_time(model, data_loader, device=torch.device('cuda')):
    model.eval()
    inference_times = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            start_time = time.time()
            _ = model(data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Asegurarse de que todas las operaciones se completen en la GPU
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
    return inference_times

##################################################################################

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import time



wandb.init(project="treball-fi-grau")

def train_model(model, optimizer, loss_fn, train_loader, val_loader, epochs):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    val_precisions, val_recalls, val_f1_scores = [], [], []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_accuracy = 0, 0
        train_loop = tqdm(train_loader, unit="batches")
        for images, labels in train_loop:
            train_loop.set_description('[TRAIN] Epoch {}/{}'.format(epoch + 1, epochs))
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)  

            loss.backward()
            optimizer.step()

            # Compute per-label losses and accuracies
            with torch.no_grad():
                label_accuracies = ((torch.sigmoid(outputs).round() == labels).float()).cpu().numpy()

                train_loss += loss.item() * images.size(0)
                train_accuracy += label_accuracies.sum(axis=0)

            train_loop.set_postfix(loss=train_loss / len(train_loader.dataset))

        train_losses.append(train_loss / len(train_loader.dataset))
        train_accuracies.append(train_accuracy / len(train_loader.dataset))

        # Validation
        model.eval()
        val_loss, val_accuracy = 0, 0
        val_predictions, val_targets = [], []
        val_loop = tqdm(val_loader, unit="batches")
        with torch.no_grad():
            for images, labels in val_loop:
                val_loop.set_description('[VAL] Epoch {}/{}'.format(epoch + 1, epochs))
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                # Compute per-label accuracies
                label_accuracies = ((torch.sigmoid(outputs).round() == labels).float()).cpu().numpy()

                val_loss += loss.item() * images.size(0)
                val_accuracy += label_accuracies.sum(axis=0)
                
                val_predictions.extend(torch.sigmoid(outputs).round().cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

                val_loop.set_postfix(loss=val_loss / len(val_loader.dataset))

        val_losses.append(val_loss / len(val_loader.dataset))
        val_accuracies.append(val_accuracy / len(val_loader.dataset))

        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        # Calculate precision, recall, and f1-score for each label
        val_precision = precision_score(val_targets, val_predictions, average=None)
        val_recall = recall_score(val_targets, val_predictions, average=None)
        val_f1 = f1_score(val_targets, val_predictions, average=None)
        
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1)

        
        # Calculate precision-recall curves for each label
        for i, (precision, recall) in enumerate(zip(val_precision, val_recall)):
            precision, recall, _ = precision_recall_curve(val_targets[:, i], val_predictions[:, i])
            plt.plot(recall, precision, marker='.')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            img_path = "precision_recall_curve.png"
            plt.savefig(img_path, format='png')
            wandb.log({f"Precision-Recall Curve_{i}": wandb.Image(img_path)})    
            plt.close()

        for i, (precision, recall, f1) in enumerate(zip(val_precision, val_recall, val_f1)):
            wandb.log({
                f"val_precision_label_{i}": precision,
                f"val_recall_label_{i}": recall,
                f"val_f1_score_label_{i}": f1
            })
            
    inference_times = inference_time(model, val_loader)
    average_inference_time = sum(inference_times) / len(inference_times)
    total_inference_time = sum(inference_times)
    average_inference_time_per_image = average_inference_time / val_loader.batch_size

    wandb.log({
        "Total inference time": total_inference_time,
        "Average inference time": average_inference_time,
        "Average inference time per image": average_inference_time_per_image
    })

    return train_losses, train_accuracies, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores


########################### Training time ######################################

start_time = time.time()

epochs = 70
train_losses, train_accuracies, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores = train_model(model, optimizer, loss_fn, train_loader, val_loader, epochs)

end_time = time.time()
training_time = end_time - start_time

print(f'Total training time:  {training_time } seconds')

#################################################################################

# Viewing results with wandb
for epoch in range(len(train_accuracies)):
    # Log average loss and accuracy
    wandb.log({
        "training_loss": train_losses[epoch],
        "validation_loss": val_losses[epoch],
        "training_accuracy": train_accuracies[epoch].mean(),
        "validation_accuracy": val_accuracies[epoch].mean()
    })
    
    # Log per-label loss and accuracy
    for i, (train_acc, val_acc) in enumerate(zip(train_accuracies[epoch], val_accuracies[epoch])):
        wandb.log({
            f"training_accuracy_label_{i}": train_acc,
            f"validation_accuracy_label_{i}": val_acc,
            f"training_loss_label_{i}": train_losses[epoch],
            f"validation_loss_label_{i}": val_losses[epoch]
        })

wandb.finish()
