import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cv2
from PIL import Image
from detection import DefectDataset

class MaskVisualizer:
    def __init__(self, dataset_path, image_dir=None):
        self.dataset_path = dataset_path
        self.image_dir = image_dir

        self.data = torch.load(dataset_path, map_location='cpu', weights_only=False)
        self.predicted_masks = self.data['predicted_masks']
        self.ground_truth_masks = self.data['ground_truth_masks']
        self.image_names = self.data['image_names']

        self.class_names = {
            0: 'Defect_1',
            1: 'Defect_2',
            2: 'Defect_3',
            3: 'Defect_4'
        }

        self.colors = ['red', 'green', 'blue', 'yellow']
        self.num_classes = 4

        print(f"Loaded {len(self.image_names)} samples.")

    def load_original_image(self, image_name):
        if not self.image_dir:
            return None

        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in extensions:
            image_path = os.path.join(self.image_dir, f"{image_name}{ext}")
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                return np.array(image)
        return None

    def create_colored_mask(self, masks, alpha=0.6, is_ground_truth=False):
        height, width = masks.shape[1], masks.shape[2]
        colored_mask = np.zeros((height, width, 4))

        num_classes_to_process = min(masks.shape[0], self.num_classes)

        for class_idx in range(num_classes_to_process):
            mask = masks[class_idx].numpy()
            if is_ground_truth:
                if mask.max() <= 1.0 and len(np.unique(mask)) <= 2:
                    mask_normalized = (mask > 0.5).astype(float)
                else:
                    mask_normalized = (mask / mask.max() if mask.max() > 0 else mask).astype(float)
                    mask_normalized = (mask_normalized > 0.5).astype(float)
            else:
                mask_normalized = (mask > 0.5).astype(float)

            if mask_normalized.max() > 0:
                color = plt.cm.tab10(class_idx)[:3]
                for c in range(3):
                    colored_mask[:, :, c] = np.maximum(
                        colored_mask[:, :, c],
                        mask_normalized * color[c]
                    )
                colored_mask[:, :, 3] = np.maximum(
                    colored_mask[:, :, 3],
                    mask_normalized * alpha
                )

        return colored_mask

    def visualize_single_sample(self, index, save_path=None, show_plot=True):
        if index >= len(self.image_names):
            print(f"Index {index} out of range.")
            return

        image_name = self.image_names[index]
        pred_masks = self.predicted_masks[index]
        gt_masks = self.ground_truth_masks[index]

        if pred_masks.shape[0] > self.num_classes:
            pred_masks = pred_masks[:self.num_classes]
        if gt_masks.shape[0] > self.num_classes:
            gt_masks = gt_masks[:self.num_classes]

        pred_colored = self.create_colored_mask(pred_masks, is_ground_truth=False)
        gt_colored = self.create_colored_mask(gt_masks, is_ground_truth=True)

        original_image = self.load_original_image(image_name)

        if original_image is not None:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()

        if original_image is not None:
            axes[0].imshow(original_image)
            axes[0].set_title(f'Original Image: {image_name}')
            axes[0].axis('off')
            start_idx = 1
        else:
            start_idx = 0

        if original_image is not None:
            overlay_pred = original_image.copy().astype(float) / 255.0
            axes[start_idx].imshow(overlay_pred)
            axes[start_idx].imshow(pred_colored)
            axes[start_idx].set_title('Prediction (Overlay)')
        else:
            axes[start_idx].imshow(pred_colored)
            axes[start_idx].set_title('Prediction')
        axes[start_idx].axis('off')

        if original_image is not None:
            overlay_gt = original_image.copy().astype(float) / 255.0
            axes[start_idx + 1].imshow(overlay_gt)
            axes[start_idx + 1].imshow(gt_colored)
            axes[start_idx + 1].set_title('Ground Truth (Overlay)')
        else:
            axes[start_idx + 1].imshow(gt_colored)
            axes[start_idx + 1].set_title('Ground Truth')
        axes[start_idx + 1].axis('off')

        axes[start_idx + 2].imshow(pred_colored)
        axes[start_idx + 2].set_title('Prediction Masks')
        axes[start_idx + 2].axis('off')

        if original_image is not None:
            axes[start_idx + 3].imshow(gt_colored)
            axes[start_idx + 3].set_title('Ground Truth Masks')
            axes[start_idx + 3].axis('off')

        legend_elements = []
        for i, (class_idx, class_name) in enumerate(self.class_names.items()):
            if i < self.num_classes:
                color = plt.cm.tab10(i)
                legend_elements.append(patches.Patch(color=color, label=class_name))

        if original_image is not None and len(axes) > 5:
            axes[5].legend(handles=legend_elements, loc='center')
            axes[5].axis('off')
        else:
            fig.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def visualize_class_statistics(self):
        pred_stats = {class_name: 0 for class_name in self.class_names.values()}
        gt_stats = {class_name: 0 for class_name in self.class_names.values()}

        for pred_masks, gt_masks in zip(self.predicted_masks, self.ground_truth_masks):
            num_classes_to_process = min(pred_masks.shape[0], self.num_classes)
            for class_idx in range(num_classes_to_process):
                class_name = self.class_names[class_idx]
                if pred_masks[class_idx].max() > 0.5:
                    pred_stats[class_name] += 1
                if gt_masks[class_idx].max() > 0.5:
                    gt_stats[class_name] += 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        classes = list(pred_stats.keys())
        pred_counts = list(pred_stats.values())
        ax1.bar(classes, pred_counts, color=self.colors)
        ax1.set_title('Prediction Samples per Class')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)

        gt_counts = list(gt_stats.values())
        ax2.bar(classes, gt_counts, color=self.colors)
        ax2.set_title('Ground Truth Samples per Class')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

        no_defect_pred = 0
        no_defect_gt = 0
        for pred_masks, gt_masks in zip(self.predicted_masks, self.ground_truth_masks):
            pred_has_defect = any(pred_masks[i].max() > 0.5 for i in range(min(pred_masks.shape[0], self.num_classes)))
            gt_has_defect = any(gt_masks[i].max() > 0.5 for i in range(min(gt_masks.shape[0], self.num_classes)))
            if not pred_has_defect:
                no_defect_pred += 1
            if not gt_has_defect:
                no_defect_gt += 1

        print(f"No defect images: Predicted {no_defect_pred}, Ground Truth {no_defect_gt}")

    def visualize_batch(self, start_index=0, batch_size=4, save_dir=None):
        end_index = min(start_index + batch_size, len(self.image_names))
        for i in range(start_index, end_index):
            print(f"Processing {i}: {self.image_names[i]}")
            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{self.image_names[i]}_comparison.png")
            self.visualize_single_sample(i, save_path=save_path, show_plot=False)
        print(f"Batch done: {end_index - start_index} samples.")


def main():
    dataset_path = './data/output/defect_dataset.pt'
    image_dir = './data/train/test_images_split'
    output_dir = './data/output/visualizations'

    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    visualizer = MaskVisualizer(dataset_path, image_dir)
    print("Class statistics...")
    visualizer.visualize_class_statistics()
    print("Batch visualizing...")
    visualizer.visualize_batch(
        start_index=0,
        batch_size=len(visualizer.image_names),
        save_dir=output_dir
    )
    print("All visualizations completed.")


if __name__ == '__main__':
    main()