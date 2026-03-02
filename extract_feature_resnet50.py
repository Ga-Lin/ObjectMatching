import os
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

def get_feature_extractor():
  # Load ResNet50
  resnet50 = models.resnet50(weights='DEFAULT')
  # Remove the final fully connected layer
  resnet50_feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
  resnet50_feature_extractor.eval()

  return resnet50_feature_extractor

def extract_feature(feature_extractor, image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # ResNet50 input: 224 * 224 RGB image
  image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)

  # Transform to tensor and normalize
  preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  return feature_extractor(torch.unsqueeze(preprocess(image), 0))


# Extract feature from gallery
gallery_dir = './gallery'
gallery_feature_dir = './gallery_features'

resnet50_feature_extractor = get_feature_extractor()
for img_file in tqdm(os.listdir(gallery_dir)):
  img_path = os.path.join(gallery_dir, img_file)
  gallery_img = cv2.imread(img_path)
  feature = extract_feature(resnet50_feature_extractor, gallery_img)
  feature_path = os.path.join(gallery_feature_dir, img_file.split('.')[0] + '.npy')
  np.save(feature_path, feature.cpu().detach().numpy())

# Extract feature from query
query_dir = './query'
query_box_dir = './query_boxes'
query_feature_dir = './query_features'

for img_file in tqdm(os.listdir(query_dir)):
  query_img_path = os.path.join(query_dir, img_file)
  query_img = cv2.imread(query_img_path)
  query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)  # BGR to RGB
  query_box_path = os.path.join(query_box_dir, img_file.split('.')[0]+'.txt')

  # Extract feature with bounding boxes
  feature = []
  for query_box in np.atleast_2d(np.loadtxt(query_box_path, dtype=int)):
    # Bounding box in format: x of top-left point, y of top-left point, width, height
    cropped_query_img = query_img[query_box[1]:query_box[1] + query_box[3],
                        query_box[0]:query_box[0] + query_box[2], :]
    sub_feature = extract_feature(resnet50_feature_extractor, cropped_query_img).cpu().detach().numpy()
    feature.append(sub_feature)

  feature_path = os.path.join(query_feature_dir, img_file.split('.')[0]+'.npy')
  np.save(feature_path, feature)