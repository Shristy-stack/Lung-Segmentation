import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
import urllib.request
from torch.utils.tensorboard import SummaryWriter
import datetime
import torchvision.transforms.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import sklearn
from sklearn.model_selection import train_test_split

# Add the project directory to the Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from segmentation.data_loader import LungSegmentationDataset  # Ensure this import path is correct
from segmentation.unet_model import UNet  # Ensure this import path is correct

class Augment:
    def __call__(self, image, mask):
        if torch.rand(1) > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        if torch.rand(1) > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)
        if torch.rand(1) > 0.5:
            angle = torch.randint(-30, 30, (1,)).item()
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)
        if torch.rand(1) > 0.5:
            image = F.adjust_brightness(image, brightness_factor=torch.rand(1).item() + 0.5)
        if torch.rand(1) > 0.5:
            image = F.adjust_contrast(image, contrast_factor=torch.rand(1).item() + 0.5)
        return image, mask

def dice_loss(preds, targets, smooth=1e-6):
    preds = preds.contiguous()
    targets = targets.contiguous()    
    intersection = (preds * targets).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (preds.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2) + smooth)
    return 1 - dice.mean()

def dice_score(preds, targets, smooth=1e-6):
    preds = preds.contiguous()
    targets = targets.contiguous()
    intersection = (preds * targets).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (preds.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2) + smooth)
    return dice.mean()

def combined_loss(preds, targets, bce_weight=0.5):
    bce = nn.BCEWithLogitsLoss()(preds, targets)
    dice = dice_loss(torch.sigmoid(preds), targets)
    return bce * bce_weight + dice * (1 - bce_weight)

def download_weights(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading pretrained weights from {url}...")
        urllib.request.urlretrieve(url, destination)
        print("Download completed.")
    else:
        print("Pretrained weights already exist.")

class LungSegmentationModel(pl.LightningModule):
    def __init__(self, pretrained_path=None, learning_rate=0.001, bce_weight=0.5):
        super(LungSegmentationModel, self).__init__()
        self.model = UNet(encoder_name='resnet50', pretrained=True, pretrained_path=pretrained_path)
        self.learning_rate = learning_rate
        self.bce_weight = bce_weight

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = combined_loss(outputs, masks, self.bce_weight)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = combined_loss(outputs, masks, self.bce_weight)
        preds = torch.sigmoid(outputs) > 0.5
        dice = dice_score(preds, masks)
        self.log('val_loss', loss)
        self.log('val_dice', dice)
        return {'val_loss': loss, 'val_dice': dice}

    def test_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = combined_loss(outputs, masks, self.bce_weight)
        preds = torch.sigmoid(outputs) > 0.5
        dice = dice_score(preds, masks)
        self.log('test_loss', loss)
        self.log('test_dice', dice)
        return {'test_loss': loss, 'test_dice': dice}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

def visualize_sample(image, mask, prediction):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image.permute(1, 2, 0).cpu())
    axs[0].set_title('Image')
    axs[1].imshow(mask.squeeze().cpu(), cmap='gray')
    axs[1].set_title('Ground Truth Mask')
    axs[2].imshow(prediction.squeeze().cpu(), cmap='gray')
    axs[2].set_title('Predicted Mask')
    plt.show()

def main(image_dir, mask_dir, pretrained_path=None, num_epochs=20, batch_size=4, learning_rate=0.001):
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory '{image_dir}' not found")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory '{mask_dir}' not found")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    augmentation = Augment()

    dataset = LungSegmentationDataset(image_dir, mask_dir, transform=transform, augmentation=augmentation)
    
    # Split dataset into training, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LungSegmentationModel(pretrained_path=pretrained_path, learning_rate=learning_rate)
    
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = TensorBoardLogger("tb_logs", name=log_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_dice',
        mode='max',
        save_top_k=1,
        verbose=True,
        filename='{epoch:02d}-{val_dice:.2f}'
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        devices=1 if torch.cuda.is_available() else None, 
        accelerator='gpu' if torch.cuda.is_available() else 'cpu', 
        logger=logger, 
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(model, train_loader, val_loader)

    # Load the best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    model = LungSegmentationModel.load_from_checkpoint(best_model_path)

    # Test the model
    trainer.test(model, test_loader)

    print('Finished Training')

if __name__ == '__main__':
    image_dir = "/home/hpc/iwi5/iwi5208h/my_project/segmentation/data/images"
    mask_dir = "/home/hpc/iwi5/iwi5208h/my_project/segmentation/data/masks"
    pretrained_path = None  # Set this to the path of your pretrained weights if available

    main(image_dir, mask_dir, pretrained_path, num_epochs=20, batch_size=4, learning_rate=0.001)
