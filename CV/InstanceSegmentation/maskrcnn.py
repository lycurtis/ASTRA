import os
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A #augmentations
from albumentations.pytorch import ToTensorV2 #convert to tensor
from torch.utils.data import Dataset

#Load the CamVid dataset (each image must have a corresponding segmentation mask)
class CamVidDataset(Dataset):
  """
  CamVid Dataset Folder:
    test (raw images)
    test_labels (segmentation masks)
    train (raw images)
    train_labels (segmentation masks)
    val (images)
    val_labels (segmentation masks)
  """
  def __init__(self, img_dir, split='train'):
        self.img_dir = img_dir
        self.split = split

        #Path Setup
        self.image_dir = os.path.join(img_dir, split)
        self.label_dir = os.path.join(img_dir, f"{split}_labels")

        self.images = os.listdir(self.image_dir)

        #Load class dictionay
        self.class_dict = pd.read_csv(os.path.join(img_dir, 'class_dict.csv'))
        """ In other words mapping specific labels to a specific color
        self.class_dict = {
          name,r,g,b
          Animal,64,128,64
          Archway,192,0,128
          Bicyclist,0,128, 192
          Bridge,0, 128, 64
          Building,128, 0, 0
          Car,64, 0, 128
          CartLuggagePram,64, 0, 192
          Child,192, 128, 64
          Column_Pole,192, 192, 128
          Fence,64, 64, 128
          LaneMkgsDriv,128, 0, 192
          LaneMkgsNonDriv,192, 0, 64
          Misc_Text,128, 128, 64
          MotorcycleScooter,192, 0, 192
          OtherMoving,128, 64, 64
          ParkingBlock,64, 192, 128
          Pedestrian,64, 64, 0
          Road,128, 64, 128
          RoadShoulder,128, 128, 192
          Sidewalk,0, 0, 192
          SignSymbol,192, 128, 128
          Sky,128, 128, 128
          SUVPickupTruck,64, 128,192
          TrafficCone,0, 0, 64
          TrafficLight,0, 64, 64
          Train,192, 64, 128
          Tree,128, 128, 0
          Truck_Bus,192, 128, 192
          Tunnel,64, 0, 64
          VegetationMisc,192, 192, 0
          Void,0, 0, 0
          Wall,64, 192, 0
        }
        """

        #create mapping of RGB colors to class indicies
        self.color_mapping = {}
        for idx, row in self.class_dict.iterrows():
          rgb = (row['r'], row['g'], row['b'])
          self.color_mapping[rgb] = idx

        #Apply data preprocessing and augmentation
        if (split == 'train'):
          # augmentation
          self.transform = A.Compose([
              A.Resize(360, 480), #Resize image and masks to standard size 3:4 aspect ratio
              A.HorizontalFlip(p=0.5), #probability of 50%
              A.RandomBrightnessContrast(p=0.2), #probability of 20%
              A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
              ToTensorV2() #Convert to Tensor
              ])
        else: #validation or test (only necessary to resize and normalize the data)
          self.transform = A.Compose([
              A.Resize(360, 480), #Resize image and masks to standard size 3:4 aspect ratio
              A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
              ToTensorV2() #Convert to Tensor
          ])

  def __len__(self):
    return len(self.images)

  #NOTE!: MASK AND LABEL ARE CONCEPTUALLY THE SAME
  def __getitem__(self, idx):
      img_name = self.images[idx]
      img_path = os.path.join(self.image_dir, img_name) #Get image path
      mask_name = img_name.replace('.jpg', '_L.png')
      mask_path = os.path.join(self.label_dir, mask_name) #Get mask/label path

      # Load image and mask
      image = np.array(Image.open(img_path).convert('RGB'))
      mask = np.array(Image.open(mask_path).convert('RGB'))

      #Convert RGB fingerprints to class indicies
      """
      each class is represented by RGB values
      e.g. Road = (128, 64, 128)
      e.g. Animal = (64,128,64)

      now instead for example
      background = 0
      road = 1
      etc...
      This conversion is necessary for training
      """
      label = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
      for rgb, idx in self.color_mapping.items():
        label[(mask[:,:,0] == rgb[0]) & (mask[:,:,1] == rgb[1]) & (mask[:,:,2] == rgb[2])] = idx

      #apply transformations so
      augment = self.transform(image=image, mask=label)
      image = augment['image']
      label = augment['mask']

      return image, label

  def calculate_iou(pred_mask, true_mask, num_classes): #Intersection over union
  IoU_scores = [] #IoUs per class or class IoUs

  for class_id in range(num_classes):
    #create binary masks
    pred_binary = (pred_mask == class_id)
    true_binary = (true_mask == class_id)

    intersection = torch.logical_and(pred_binary, true_binary).sum()
    union = torch.logical_or(pred_binary, true_binary).sum()

    e = 1e-8 #necessary (epsilon) small value for safety measures to avoid undefined behavior
    IoU = intersection.float()/ (union.float() + e) #small value to avoid division by zero
    IoU_scores.append(IoU.item()) #.item() to convert a single-element PyTorch tensor to a python number
  return np.mean(IoU_scores)

def pixel_accuracy(pred_mask, true_mask):
  correct_pixels = (pred_mask == true_mask).sum()
  total_pixels = torch.numel(true_mask)
  return correct_pixels.float()/total_pixels
