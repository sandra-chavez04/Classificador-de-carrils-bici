import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import torchvision
import matplotlib.pyplot as plt
import os, shutil
import wandb
from PIL import Image
from torchvision.models import resnet50
from torch.utils.data import TensorDataset

if not torch.cuda.is_available():
    raise Exception("You should enable GPU in the Runtime menu")
device = torch.device("cuda")

api_key = "your_wandb api key"
wandb.login(key=api_key)


############################################################################################################
                                    ## Image Organization ##
############################################################################################################

# The path to the directory where the original data set is located
original_dataset_dir_unidirectional = 'route where the /LaneAnalysis/binary_classifier/unidirectional_yes folder is located'
original_dataset_dir_no_unidirectional = 'route where the /LaneAnalysis/binary_classifier/unidirectional_no folder is located'

# The directory where we will store our smallest data set
base_dir = './processed_data'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# Directories for our training and validation splits
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)

# Directory with our training unidirectional pictures
train_unidirectional_dir = os.path.join(train_dir, 'unidirectional')
if not os.path.exists(train_unidirectional_dir):
    os.mkdir(train_unidirectional_dir)

# Directory with our training no unidirectional pictures
train_no_unidirectional_dir = os.path.join(train_dir, 'no_unidirectional')
if not os.path.exists(train_no_unidirectional_dir):
    os.mkdir(train_no_unidirectional_dir)

# Directory with our validation unidirectional pictures
validation_unidirectional_dir = os.path.join(validation_dir, 'unidirectional')
if not os.path.exists(validation_unidirectional_dir):
    os.mkdir(validation_unidirectional_dir)

# Directory with our validation no unidirectional pictures
validation_no_unidirectional_dir = os.path.join(validation_dir, 'no_unidirectional')
if not os.path.exists(validation_no_unidirectional_dir):
    os.mkdir(validation_no_unidirectional_dir)


# Copy 70% of the unidirectional images to train_unidirectional_dir and 30% to validation_unidirectional_dir
fnames = os.listdir(original_dataset_dir_unidirectional)
random.shuffle(fnames)
fnames_to_copy = fnames[:int(0.7 * len(fnames))]

for i, fname in enumerate(fnames_to_copy):
    src = os.path.join(original_dataset_dir_unidirectional, fname)
    dst = os.path.join(train_unidirectional_dir, "unidirectional{}.jpg".format(i+1))
    shutil.copyfile(src, dst)

for i, fname in enumerate(fnames[int(0.7 * len(fnames)):]):
    src = os.path.join(original_dataset_dir_unidirectional, fname)
    dst = os.path.join(validation_unidirectional_dir, "unidirectional{}.jpg".format(i+len(fnames_to_copy)+1))
    shutil.copyfile(src, dst)

# Copy 70% of the no unidirectional images to train_no_unidirectional_dir and 30% to validation_no_unidirectional_dir
fnames = os.listdir(original_dataset_dir_no_unidirectional)
random.shuffle(fnames)
fnames_to_copy = fnames[:int(0.7 * len(fnames))]

for i, fname in enumerate(fnames_to_copy):
    src = os.path.join(original_dataset_dir_no_unidirectional, fname)
    dst = os.path.join(train_no_unidirectional_dir, "no_unidirectional{}.jpg".format(i+1))
    shutil.copyfile(src, dst)

for i, fname in enumerate(fnames[int(0.7 * len(fnames)):]):
    src = os.path.join(original_dataset_dir_no_unidirectional, fname)
    dst = os.path.join(validation_no_unidirectional_dir, "no_unidirectional{}.jpg".format(i+len(fnames_to_copy)+1))
    shutil.copyfile(src, dst)

# Image count we have in each training division
print('total training unidirectional images:', len(os.listdir(train_unidirectional_dir)))
print('total training no unidirectional images:', len(os.listdir(train_no_unidirectional_dir)))
print('total validation unidirectional images:', len(os.listdir(validation_unidirectional_dir)))
print('total validation no unidirectional images:', len(os.listdir(validation_no_unidirectional_dir)))


############################################################################################################
                           ## Apply data augmentation to balance folders ##
############################################################################################################

# Define training and validation transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # Random horizontal flip
    transforms.RandomRotation(10),          # Random rotation in a range of -10 to 10 degrees
    transforms.ToTensor()                   # Convert the image to a tensor
])
val_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # Random horizontal flip
    transforms.RandomRotation(10),          # Random rotation in a range of -10 to 10 degrees
    transforms.ToTensor()                   # Convert the image to a tensor
])

count_unidirectional_train = len(os.listdir(train_unidirectional_dir))
count_no_unidirectional_train = len(os.listdir(train_no_unidirectional_dir))

# Calculate the difference between the number of images of the two classes in the training folder.
diff_unidirectional_train = len(os.listdir(train_no_unidirectional_dir)) - len(os.listdir(train_unidirectional_dir))

# Apply data augmentation to the class with fewer samples
if count_unidirectional_train < count_no_unidirectional_train:
    dir_to_augment = train_unidirectional_dir
else:
    dir_to_augment = train_no_unidirectional_dir

for i in range(abs(diff_unidirectional_train)):
    img_name = os.listdir(dir_to_augment)[random.randint(0, len(os.listdir(dir_to_augment)) - 1)]
    img = Image.open(os.path.join(dir_to_augment, img_name))
    img = train_transform(img)
    img = img.unsqueeze(0)
    img_name = "unidirectional_augmented_" + str(i+1) + ".jpg"
    img_path = os.path.join(dir_to_augment, img_name)
    torchvision.utils.save_image(img, img_path)

count_unidirectional_val = len(os.listdir(validation_unidirectional_dir))
count_no_unidirectional_val = len(os.listdir(validation_no_unidirectional_dir))

# Calculate the difference between the number of images of the two classes in the validation folder.
diff_unidirectional_val = len(os.listdir(validation_no_unidirectional_dir)) - len(os.listdir(validation_unidirectional_dir))

# Apply data augmentation to the class with fewer samples
if count_unidirectional_val < count_no_unidirectional_val:
    dir_to_augment = validation_unidirectional_dir
else:
    dir_to_augment = validation_no_unidirectional_dir

for i in range(abs(diff_unidirectional_val)):
    img_name = os.listdir(dir_to_augment)[random.randint(0, len(os.listdir(dir_to_augment)) - 1)]
    img = Image.open(os.path.join(dir_to_augment, img_name))
    img = val_transform(img)
    img = img.unsqueeze(0)
    img_name = "unidirectional_augmented_" + str(i+1) + ".jpg"
    img_path = os.path.join(dir_to_augment, img_name)
    torchvision.utils.save_image(img, img_path)

# Count how many images we have again in each training split
print('total training unidirectional images after data augmentation:', len(os.listdir(train_unidirectional_dir)))
print('total training no unidirectional images after data augmentation:', len(os.listdir(train_no_unidirectional_dir)))
print('total validation unidirectional images after data augmentation:', len(os.listdir(validation_unidirectional_dir)))
print('total validation no unidirectional images after data augmentation:', len(os.listdir(validation_no_unidirectional_dir)))


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
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 1),  
    nn.Sigmoid()  # Use Sigmoid for binary classification
)

model = nn.Sequential(
            feature_extractor,
            nn.Flatten(),
            feature_classifier
)

model.to(device)


############################# Required memory ####################################

import torchsummary  # type: ignore

torchsummary.summary(model, input_size=(3, 224, 224))

##################################################################################

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
augmented_dataset = ImageFolder(train_dir, transform=train_transform)
augmented_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)
val_dataset = ImageFolder(validation_dir, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCELoss()


############################ Inference time ####################################

import time

def inference_time(model, data_loader, device=torch.device('cuda')):
    model.eval()
    inference_times = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            start_time = time.time()
            output = model(data)
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
    return inference_times

#################################################################################

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_model(model, optimizer, loss_fn, train_loader, val_loader, epochs, device=torch.device('cuda')):
    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    val_precision, val_recall, val_f1 = [], [], []

    # Initialize WandB
    wandb.init(project="treball-fi-grau")

    for epoch in range(epochs):
        # train
        model.train()
        train_loss.reset()
        train_accuracy.reset()
        train_loop = tqdm(train_loader, unit=" batches")  # For printing the progress bar
        for data, target in train_loop:
            train_loop.set_description('[TRAIN] Epoch {}/{}'.format(epoch + 1, epochs))
            data, target = data.float().to(device), target.float().to(device)
            target = target.unsqueeze(-1)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), n=len(target))
            pred = output.round()  # get the prediction
            acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
            train_accuracy.update(acc, n=len(target))
            train_loop.set_postfix(loss=train_loss.avg, accuracy=train_accuracy.avg)

        train_losses.append(train_loss.avg)
        train_accuracies.append(train_accuracy.avg)

        # validation
        model.eval()
        val_loss.reset()
        val_accuracy.reset()
        val_loop = tqdm(val_loader, unit=" batches")  # For printing the progress bar
        with torch.no_grad():
            preds = []
            targets = []
            for data, target in val_loop:
                val_loop.set_description('[VAL] Epoch {}/{}'.format(epoch + 1, epochs))
                data, target = data.float().to(device), target.float().to(device)
                target = target.unsqueeze(-1)
                output = model(data)
                loss = loss_fn(output, target)
                val_loss.update(loss.item(), n=len(target))
                pred = output.round()  # get the prediction
                acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
                val_accuracy.update(acc, n=len(target))
                val_loop.set_postfix(loss=val_loss.avg, accuracy=val_accuracy.avg)

                # Save predictions and targets for precision, recall, and F1-score calculation
                preds.extend(pred.cpu().detach().numpy())
                targets.extend(target.cpu().detach().numpy())

            val_losses.append(val_loss.avg)
            val_accuracies.append(val_accuracy.avg)

            # Calculate precision, recall, and F1-score
            precision = precision_score(targets, preds)
            recall = recall_score(targets, preds)
            f1 = f1_score(targets, preds)
            val_precision.append(precision)
            val_recall.append(recall)
            val_f1.append(f1)

            
            # Log precision, recall, and F1-score to WandB
            wandb.log({
                "Validation Precision": precision,
                "Validation Recall": recall,
                "Validation F1 Score": f1
            })

    # Plot the precision-recall curve
    precision, recall, _ = precision_recall_curve(targets, preds)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    img_path = "precision_recall_curve.png"
    plt.savefig(img_path, format='png')
    wandb.log({"Precision-Recall Curve": wandb.Image(img_path)})    
    plt.close()

    # Calculate inference times
    inference_times = inference_time(model, val_loader)
    average_inference_time = sum(inference_times) / len(inference_times)
    total_inference_time = sum(inference_times)
    
    # Log inference times to WandB
    wandb.log({
        "Total Inference Time": total_inference_time,
        "Average Inference Time": average_inference_time
    })

    print("Total Inference Time:", total_inference_time)
    print("Average Inference Time:", average_inference_time)

    return train_accuracies, train_losses, val_accuracies, val_losses, val_precision, val_recall, val_f1

########################### Training time ######################################

start_time = time.time()

epochs = 10
train_accuracies, train_losses, val_accuracies, val_losses, val_precision, val_recall, val_f1 = train_model(
    model, optimizer, loss_fn, augmented_loader, val_loader, epochs)

end_time = time.time()

total_training_time_seconds = end_time - start_time
print(f'Total training time:  {total_training_time_seconds } seconds')

#################################################################################                      

wandb.finish()

# Viewing results with wandb
wandb.init(project="treball-fi-grau")

for epoch in range(len(train_accuracies)):
    wandb.log({
        "training_accuracy": train_accuracies[epoch],
        "validation_accuracy": val_accuracies[epoch],
        "training_loss": train_losses[epoch],
        "validation_loss": val_losses[epoch]
        
    })
 
wandb.finish()
