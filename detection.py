import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import get_model
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import shutil

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def rle_to_mask(rle_string, height, width):
    if pd.isna(rle_string) or rle_string == '' or rle_string == 'no_defect':
        return np.zeros((height, width), dtype=np.uint8)
    try:
        rle = list(map(int, rle_string.split()))
        starts = rle[0::2]
        lengths = rle[1::2]
        mask = np.zeros(height * width, dtype=np.uint8)
        for start, length in zip(starts, lengths):
            start_idx = start - 1
            end_idx = start_idx + length
            if start_idx >= 0 and end_idx <= height * width:
                mask[start_idx:end_idx] = 1
        mask = mask.reshape((width, height)).T
    except Exception as e:
        print(f"RLE decode error: {e}")
        return np.zeros((height, width), dtype=np.uint8)
    return mask

def get_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return height, width
    except Exception as e:
        print(f"Cannot read image size {image_path}: {e}")
        return 256, 1600

def split_train_test(csv_path, train_images_dir, test_split_ratio=0.2, random_state=42):
    print("Splitting train and test sets...")
    df = pd.read_csv(csv_path)
    image_names = df['ImageId'].unique()
    train_images, test_images = train_test_split(
        image_names,
        test_size=test_split_ratio,
        random_state=random_state
    )
    test_images_dir = train_images_dir.replace('train_images', 'test_images_split')
    os.makedirs(test_images_dir, exist_ok=True)
    copied_count = 0
    for image_name in test_images:
        src_path = os.path.join(train_images_dir, image_name)
        dst_path = os.path.join(test_images_dir, image_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
    print(f"Copied {copied_count} test images to {test_images_dir}")
    test_df = df[df['ImageId'].isin(test_images)].copy()
    test_csv_path = csv_path.replace('.csv', '_test.csv')
    test_df.to_csv(test_csv_path, index=False)
    return test_images_dir, test_csv_path, test_images

class DefectDataset(Dataset):
    def __init__(self, predicted_masks, ground_truth_masks, image_names):
        self.predicted_masks = predicted_masks
        self.ground_truth_masks = ground_truth_masks
        self.image_names = image_names
        assert len(predicted_masks) == len(ground_truth_masks) == len(image_names), \
            "All lists must have the same length"
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, idx):
        return {
            'image_name': self.image_names[idx],
            'predicted_mask': self.predicted_masks[idx],
            'ground_truth_mask': self.ground_truth_masks[idx]
        }

class DefectDetector:
    def __init__(self, model_path=None, num_classes=4, device=None, use_pretrained=False):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        model_num_classes = num_classes + 1 if num_classes == 4 else num_classes
        self.model = get_model(model_num_classes)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded trained model: {model_path}")
        elif use_pretrained:
            print("Using pretrained model for demo")
        else:
            print(f"Model file not found: {model_path}")
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Set use_pretrained=True to use a pretrained model.")
        self.model.to(self.device)
        self.model.eval()
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_names = {1: 'fault_1', 2: 'fault_2', 3: 'fault_3', 4: 'fault_4'}
    def load_ground_truth_data(self, csv_path):
        df = pd.read_csv(csv_path)
        return df
    def match_images_with_ground_truth(self, image_names, gt_df):
        matched_data = {}
        for image_name in image_names:
            image_id_variants = [
                f"{image_name}.jpg",
                f"{image_name}",
                image_name
            ]
            matching_rows = pd.DataFrame()
            for variant in image_id_variants:
                matching_rows = gt_df[gt_df['ImageId'] == variant]
                if len(matching_rows) > 0:
                    break
            if len(matching_rows) > 0:
                matched_data[image_name] = matching_rows
            else:
                matched_data[image_name] = pd.DataFrame({
                    'ImageId': [f"{image_name}.jpg"],
                    'ClassId': [0],
                    'EncodedPixels': ['']
                })
        return matched_data
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor, original_image
    def filter_predictions(self, predictions, confidence_threshold=0.5, mask_threshold=0.5):
        pred = predictions
        keep = pred['scores'] > confidence_threshold
        filtered_boxes = pred['boxes'][keep]
        filtered_masks = pred['masks'][keep]
        filtered_labels = pred['labels'][keep]
        filtered_scores = pred['scores'][keep]
        if len(filtered_masks) > 0:
            binary_masks = (filtered_masks > mask_threshold).float()
        else:
            binary_masks = filtered_masks
        results = {
            'boxes': filtered_boxes,
            'masks': binary_masks,
            'labels': filtered_labels,
            'scores': filtered_scores
        }
        return results
    def combine_masks_by_class(self, masks, labels, image_shape, num_classes=4):
        height, width = image_shape
        class_masks = torch.zeros((num_classes, height, width), dtype=torch.float32)
        for mask, label in zip(masks, labels):
            if mask.ndim == 3:
                mask = mask[0]
            label_idx = int(label.item()) - 1
            if 0 <= label_idx < num_classes:
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (width, height))
                mask_tensor = torch.from_numpy(mask_resized)
                class_masks[label_idx] = torch.maximum(class_masks[label_idx], mask_tensor)
        return class_masks
    def create_ground_truth_tensor(self, gt_data, image_shape, num_classes=4):
        height, width = image_shape
        gt_masks = torch.zeros((num_classes, height, width), dtype=torch.float32)
        for _, row in gt_data.iterrows():
            class_id = int(row['ClassId'])
            if class_id > 0 and class_id <= num_classes:
                rle_string = row['EncodedPixels']
                if pd.notna(rle_string) and rle_string != '':
                    mask = rle_to_mask(rle_string, height, width)
                    if mask.shape != (height, width):
                        mask = cv2.resize(mask.astype(np.float32), (width, height),
                                          interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    mask_tensor = torch.from_numpy(mask).float()
                    gt_masks[class_id - 1] = torch.maximum(gt_masks[class_id - 1], mask_tensor)
        return gt_masks
    def process_single_image(self, image_path, confidence_threshold, mask_threshold=0.5):
        try:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            actual_height, actual_width = get_image_dimensions(image_path)
            image_tensor, original_image = self.preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)
            with torch.no_grad():
                predictions = self.model(image_tensor)
            raw_results = {
                'boxes': predictions[0]['boxes'],
                'masks': predictions[0]['masks'],
                'labels': predictions[0]['labels'],
                'scores': predictions[0]['scores']
            }
            filtered_results = self.filter_predictions(
                predictions[0],
                confidence_threshold,
                mask_threshold
            )
            return image_name, {'raw': raw_results, 'filtered': filtered_results}, (actual_height, actual_width), True
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return os.path.splitext(os.path.basename(image_path))[0], None, None, False
    def process_images_to_dataset(self, image_paths, gt_csv_path, output_dir, confidence_threshold=0.5,
                                  mask_threshold=0.5):
        print(f"Processing {len(image_paths)} images...")
        gt_df = self.load_ground_truth_data(gt_csv_path)
        predicted_masks_list = []
        ground_truth_masks_list = []
        image_names_list = []
        successful_count = 0
        failed_count = 0
        for i, image_path in enumerate(image_paths):
            image_name, results, image_shape, success = self.process_single_image(
                image_path, confidence_threshold, mask_threshold
            )
            if success and results:
                matched_gt = self.match_images_with_ground_truth([image_name], gt_df)
                gt_data = matched_gt[image_name]
                filtered_results = results['filtered']
                if len(filtered_results['masks']) > 0:
                    pred_masks = self.combine_masks_by_class(
                        filtered_results['masks'],
                        filtered_results['labels'],
                        image_shape,
                        self.num_classes
                    )
                else:
                    pred_masks = torch.zeros((self.num_classes, image_shape[0], image_shape[1]), dtype=torch.float32)
                gt_masks = self.create_ground_truth_tensor(gt_data, image_shape, self.num_classes)
                predicted_masks_list.append(pred_masks)
                ground_truth_masks_list.append(gt_masks)
                image_names_list.append(image_name)
                successful_count += 1
            else:
                failed_count += 1
        print(f"Summary: {successful_count} succeeded, {failed_count} failed")
        if predicted_masks_list:
            dataset = DefectDataset(predicted_masks_list, ground_truth_masks_list, image_names_list)
            dataset_path = os.path.join(output_dir, 'defect_dataset.pt')
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'predicted_masks': predicted_masks_list,
                'ground_truth_masks': ground_truth_masks_list,
                'image_names': image_names_list,
                'dataset': dataset
            }, dataset_path)
            print(f"Dataset saved to {dataset_path}")
            return dataset
        else:
            print("No successful results, dataset not created")
            return None
    def create_dataloader(self, dataset, batch_size=8, shuffle=False, num_workers=0):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def load_dataset(dataset_path):
    data = torch.load(dataset_path)
    return data['dataset']

def main():
    model_path = 'fault_detection_maskrcnn.pth'
    train_images_dir = './data/train/train_images'
    train_csv_path = './data/severstal-steel-defect-detection/train.csv'
    output_dir = './data/output'
    confidence_threshold = 0.8
    mask_threshold = 0.5
    test_split_ratio = 0.2
    print("Splitting test set from train set...")
    test_images_dir, test_csv_path, test_images = split_train_test(
        train_csv_path,
        train_images_dir,
        test_split_ratio,
        random_state=42
    )
    if os.path.exists(model_path):
        detector = DefectDetector(model_path, num_classes=4)
    else:
        detector = DefectDetector(use_pretrained=True, num_classes=4)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    test_image_paths = []
    for image_name in test_images:
        image_path = os.path.join(test_images_dir, image_name)
        if os.path.exists(image_path) and any(image_name.lower().endswith(ext) for ext in image_extensions):
            test_image_paths.append(image_path)
    if not test_image_paths:
        print("No valid images found in test set directory.")
        return
    print(f"Found {len(test_image_paths)} test images")
    dataset = detector.process_images_to_dataset(
        test_image_paths, test_csv_path, output_dir, confidence_threshold, mask_threshold
    )
    if dataset:
        dataloader = detector.create_dataloader(dataset, batch_size=4, shuffle=True)
        print(f"Dataset created, contains {len(dataset)} samples")
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx + 1}: {len(batch['image_name'])} samples")
            if batch_idx >= 1:
                break
    else:
        print("Dataset creation failed")

if __name__ == '__main__':
    main()