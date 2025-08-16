import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import get_model
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy import ndimage
import pandas as pd
import matplotlib.cm as cm


class DefectEdgeDetector:
    def __init__(self, model_path=None, num_classes=5, device=None, use_pretrained=False):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes

        self.model = get_model(num_classes)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded trained model from {model_path}")
        elif use_pretrained:
            print("Using pretrained model for demonstration")
        else:
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Set use_pretrained=True to use pretrained model for testing.")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 故障类别名称和颜色映射
        self.class_names = {1: 'fault_1', 2: 'fault_2', 3: 'fault_3', 4: 'fault_4'}
        self.class_colors = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255], 4: [255, 255, 0]}

    def mask_to_rle(self, mask):
        """将掩码转换为RLE编码"""
        mask_flat = mask.flatten()
        mask_binary = (mask_flat > 0.5).astype(np.uint8)

        rle = []
        current_pixel = 0

        while current_pixel < len(mask_binary):
            while current_pixel < len(mask_binary) and mask_binary[current_pixel] == 0:
                current_pixel += 1

            if current_pixel >= len(mask_binary):
                break

            start = current_pixel

            while current_pixel < len(mask_binary) and mask_binary[current_pixel] == 1:
                current_pixel += 1

            length = current_pixel - start
            rle.extend([start, length])

        return ' '.join(map(str, rle)) if rle else ''

    def create_raw_heatmap(self, raw_results, original_shape):
        """基于原始检测结果创建热力图（不受阈值限制）"""
        height, width = original_shape[:2]
        heatmap = np.zeros((height, width, 3), dtype=np.float32)

        for mask, label, score in zip(raw_results['masks'], raw_results['labels'], raw_results['scores']):
            if mask.ndim == 3:
                mask = mask[0]

            mask_resized = cv2.resize(mask, (width, height))
            color = self.class_colors.get(label, [128, 128, 128])

            for c in range(3):

                channel_value = mask_resized * score * (color[c] / 255.0)
                heatmap[:, :, c] = np.maximum(heatmap[:, :, c], channel_value)

        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        return heatmap

    def create_heatmap(self, results, original_shape):
        height, width = original_shape[:2]
        heatmap = np.zeros((height, width, 3), dtype=np.float32)

        for mask, label, score in zip(results['masks'], results['labels'], results['scores']):
            if mask.ndim == 3:
                mask = mask[0]

            mask_resized = cv2.resize(mask, (width, height))

            color = self.class_colors.get(label, [128, 128, 128])  # 默认灰色

            for c in range(3):
                channel_value = mask_resized * score * (color[c] / 255.0)
                heatmap[:, :, c] = np.maximum(heatmap[:, :, c], channel_value)

        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        return heatmap

    def save_pixel_scores(self, results, original_shape, image_name, output_dir):
        """保存每张图片每个像素每个缺陷类别的分数"""
        height, width = original_shape[:2]

        pixel_scores_dir = os.path.join(output_dir, 'pixel_scores')
        os.makedirs(pixel_scores_dir, exist_ok=True)

        class_score_maps = {}
        for class_id in range(1, self.num_classes):  # 1到4类
            class_score_maps[class_id] = np.zeros((height, width), dtype=np.float32)

        if len(results['masks']) > 0:
            for mask, label, score in zip(results['masks'], results['labels'], results['scores']):
                if mask.ndim == 3:
                    mask = mask[0]

                mask_resized = cv2.resize(mask, (width, height))

                if label in class_score_maps:
                    pixel_scores = mask_resized * score
                    class_score_maps[label] = np.maximum(class_score_maps[label], pixel_scores)

        for class_id, score_map in class_score_maps.items():
            npy_filename = f"{image_name}_class_{class_id}_scores.npy"
            np.save(os.path.join(pixel_scores_dir, npy_filename), score_map)

        combined_scores = np.zeros((height, width), dtype=np.float32)
        combined_labels = np.zeros((height, width), dtype=np.int32)

        for class_id, score_map in class_score_maps.items():
            better_mask = score_map > combined_scores
            combined_scores[better_mask] = score_map[better_mask]
            combined_labels[better_mask] = class_id

        np.save(os.path.join(pixel_scores_dir, f"{image_name}_combined_scores.npy"), combined_scores)
        np.save(os.path.join(pixel_scores_dir, f"{image_name}_combined_labels.npy"), combined_labels)

        if width * height < 1000000:
            pixel_data = []
            for y in range(height):
                for x in range(width):
                    row_data = {
                        'image_name': image_name,
                        'x': x,
                        'y': y,
                        'combined_score': combined_scores[y, x],
                        'predicted_class': combined_labels[y, x]
                    }
                    for class_id in range(1, self.num_classes):
                        row_data[f'class_{class_id}_score'] = class_score_maps[class_id][y, x]

                    pixel_data.append(row_data)

            df_pixels = pd.DataFrame(pixel_data)
            csv_filename = f"{image_name}_pixel_scores.csv"
            df_pixels.to_csv(os.path.join(pixel_scores_dir, csv_filename), index=False)
            print(f"  Saved pixel scores CSV: {csv_filename}")

        print(f"  Saved pixel score maps for {image_name}")
        return class_score_maps, combined_scores, combined_labels

    def create_score_visualization(self, class_score_maps, combined_scores, combined_labels,
                                  image_name, output_dir):
        visualizations_dir = os.path.join(output_dir, 'score_visualizations')
        os.makedirs(visualizations_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, (class_id, score_map) in enumerate(class_score_maps.items()):
            if i < 4:
                im = axes[i].imshow(score_map, cmap='hot', vmin=0, vmax=1)
                axes[i].set_title(f'Class {class_id} Scores')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        if len(class_score_maps) >= 1:
            im = axes[4].imshow(combined_scores, cmap='hot', vmin=0, vmax=1)
            axes[4].set_title('Combined Scores')
            axes[4].axis('off')
            plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

        if len(class_score_maps) >= 1:
            label_colors = np.zeros((*combined_labels.shape, 3), dtype=np.uint8)
            for class_id in range(1, self.num_classes):
                mask = combined_labels == class_id
                color = self.class_colors.get(class_id, [128, 128, 128])
                label_colors[mask] = color

            axes[5].imshow(label_colors)
            axes[5].set_title('Predicted Classes')
            axes[5].axis('off')

        for i in range(len(class_score_maps) + 2, 6):
            if i < 6:
                axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_dir, f"{image_name}_score_visualization.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def save_results_to_csv(self, all_results, output_dir):
        csv_data = []

        for image_name, results in all_results.items():
            if results and len(results['masks']) > 0:
                for i, (mask, label, score, box) in enumerate(zip(
                        results['masks'], results['labels'], results['scores'], results['boxes']
                )):
                    if mask.ndim == 3:
                        mask = mask[0]

                    rle = self.mask_to_rle(mask)

                    fault_name = self.class_names.get(label, f'fault_{label}')

                    csv_data.append({
                        'image_name': image_name,
                        'defect_id': i,
                        'class_label': label,
                        'class_name': fault_name,
                        'confidence_score': score,
                        'bbox_x1': int(box[0]),
                        'bbox_y1': int(box[1]),
                        'bbox_x2': int(box[2]),
                        'bbox_y2': int(box[3]),
                        'rle_encoding': rle
                    })
            else:
                csv_data.append({
                    'image_name': image_name,
                    'defect_id': -1,
                    'class_label': 0,
                    'class_name': 'no_defect',
                    'confidence_score': 0.0,
                    'bbox_x1': 0,
                    'bbox_y1': 0,
                    'bbox_x2': 0,
                    'bbox_y2': 0,
                    'rle_encoding': ''
                })

        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, 'detection_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        return df

    def process_image(self, image_path, output_dir, confidence_threshold=0.5,
                      edge_method='canny', save_individual=True):
        # 获取过滤后的结果用于边缘检测和注释
        results, original_image = self.detect_defects(image_path, confidence_threshold)

        # 获取原始结果用于生成完整热力图
        raw_results, _ = self.detect_defects_with_raw_outputs(image_path)

        image_name = os.path.splitext(os.path.basename(image_path))[0]

        edges_dir = os.path.join(output_dir, 'edges')
        heatmaps_dir = os.path.join(output_dir, 'heatmaps')
        analysis_dir = os.path.join(output_dir, 'analysis')
        overlays_dir = os.path.join(output_dir, 'overlays')

        for dir_path in [edges_dir, heatmaps_dir, analysis_dir, overlays_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 使用过滤后的结果保存像素分数
        class_score_maps, combined_scores, combined_labels = self.save_pixel_scores(
            results, original_image.shape, image_name, output_dir
        )

        self.create_score_visualization(
            class_score_maps, combined_scores, combined_labels, image_name, output_dir
        )

        height, width = original_image.shape[:2]
        combined_edges = np.zeros((height, width), dtype=np.uint8)
        combined_masks = np.zeros((height, width), dtype=np.uint8)
        edge_results = []

        # 处理过滤后的检测结果进行边缘检测
        if len(results['masks']) > 0:
            print(f"Processing {len(results['masks'])} defects in {image_path}")

            for i, (mask, label, score, box) in enumerate(zip(
                    results['masks'], results['labels'], results['scores'], results['boxes']
            )):
                if mask.ndim == 3:
                    mask = mask[0]

                mask_resized = cv2.resize(mask, (width, height))

                if edge_method == 'canny':
                    edges = self.extract_edges_canny(mask_resized)
                elif edge_method == 'contour':
                    edges, contours = self.extract_edges_contour(mask_resized)
                elif edge_method == 'morphology':
                    edges = self.extract_edges_morphology(mask_resized)
                else:
                    raise ValueError("edge_method must be 'canny', 'contour', or 'morphology'")

                edges_refined = self.refine_edges(edges)

                combined_edges = cv2.bitwise_or(combined_edges, edges_refined)
                combined_masks = cv2.bitwise_or(combined_masks, (mask_resized * 255).astype(np.uint8))

                if save_individual:
                    fault_name = self.class_names.get(label, f'fault_{label}')
                    edge_filename = f"{image_name}_{fault_name}_{i}_edges.png"
                    cv2.imwrite(os.path.join(edges_dir, edge_filename), edges_refined)

                edge_results.append({
                    'label': label,
                    'score': score,
                    'box': box,
                    'edges': edges_refined,
                    'mask': mask_resized
                })
        else:
            print(f"No defects detected above threshold {confidence_threshold} in {image_path}")

        # 始终保存组合边缘结果
        cv2.imwrite(os.path.join(edges_dir, f"{image_name}_combined_edges.png"), combined_edges)

        # 安全检查原始结果并创建热力图
        if raw_results and 'masks' in raw_results and len(raw_results['masks']) > 0:
            raw_heatmap = self.create_raw_heatmap(raw_results, original_image.shape)
            raw_detections_count = len(raw_results['masks'])
        else:
            raw_heatmap = np.zeros_like(original_image)
            raw_detections_count = 0

        # 同时保存两种热力图
        cv2.imwrite(os.path.join(heatmaps_dir, f"{image_name}_raw_heatmap.png"),
                    cv2.cvtColor(raw_heatmap, cv2.COLOR_RGB2BGR))

        # 保存过滤后的热力图
        if len(results['masks']) > 0:
            filtered_heatmap = self.create_heatmap(results, original_image.shape)
        else:
            filtered_heatmap = np.zeros_like(original_image)

        cv2.imwrite(os.path.join(heatmaps_dir, f"{image_name}_filtered_heatmap.png"),
                    cv2.cvtColor(filtered_heatmap, cv2.COLOR_RGB2BGR))

        # 始终创建分析图表（无论是否有检测结果）
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(combined_masks, cmap='gray')
        axes[1].set_title('Detected Masks')
        axes[1].axis('off')

        axes[2].imshow(combined_edges, cmap='gray')
        axes[2].set_title(f'Edges ({edge_method})')
        axes[2].axis('off')

        # 在分析图中使用原始热力图
        axes[3].imshow(raw_heatmap)
        axes[3].set_title('Raw Heatmap (All Detections)')
        axes[3].axis('off')

        # 创建带注释的图像
        annotated = original_image.copy()
        for result in edge_results:
            box = result['box'].astype(int)
            label = result['label']
            score = result['score']
            fault_name = self.class_names.get(label, f'fault_{label}')

            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

            label_text = f"{fault_name}: {score:.2f}"
            cv2.putText(annotated, label_text, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        axes[4].imshow(annotated)
        axes[4].set_title('Detection Results')
        axes[4].axis('off')

        # 计算统计信息
        edge_pixels = np.sum(combined_edges > 0)
        total_pixels = combined_edges.shape[0] * combined_edges.shape[1]
        edge_ratio = edge_pixels / total_pixels * 100

        stats_text = f"Edge pixels: {edge_pixels}\nEdge ratio: {edge_ratio:.2f}%\nFiltered defects: {len(edge_results)}\nRaw detections: {raw_detections_count}"
        axes[5].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        axes[5].set_title('Statistics')
        axes[5].axis('off')

        plt.tight_layout()
        # 始终保存分析图表
        plt.savefig(os.path.join(analysis_dir, f"{image_name}_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 创建叠加图像（使用原始热力图）
        overlay_raw = cv2.addWeighted(original_image, 0.6, raw_heatmap, 0.4, 0)
        cv2.imwrite(os.path.join(overlays_dir, f"{image_name}_overlay_raw_heatmap.png"),
                    cv2.cvtColor(overlay_raw, cv2.COLOR_RGB2BGR))

        # 创建过滤后的叠加图像
        overlay_filtered = cv2.addWeighted(original_image, 0.6, filtered_heatmap, 0.4, 0)
        cv2.imwrite(os.path.join(overlays_dir, f"{image_name}_overlay_filtered_heatmap.png"),
                    cv2.cvtColor(overlay_filtered, cv2.COLOR_RGB2BGR))

        print(f"Analysis saved for {image_name} (Filtered: {len(edge_results)}, Raw: {raw_detections_count})")

        return results

    def preprocess_image(self, image_path):
        """预处理输入图像"""
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)

        # 转换为tensor
        image_tensor = self.transform(image).unsqueeze(0)

        return image_tensor, original_image

    def detect_defects_with_raw_outputs(self, image_path):
        """获取原始检测结果（不进行阈值过滤）"""
        image_tensor, original_image = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        pred = predictions[0]

        # 返回未过滤的原始结果
        raw_results = {
            'boxes': pred['boxes'].cpu().numpy(),
            'masks': pred['masks'].cpu().numpy(),
            'labels': pred['labels'].cpu().numpy(),
            'scores': pred['scores'].cpu().numpy()
        }

        return raw_results, original_image

    def detect_defects(self, image_path, confidence_threshold=0.5):
        image_tensor, original_image = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        pred = predictions[0]

        # 过滤低置信度的检测结果
        keep = pred['scores'] > confidence_threshold

        results = {
            'boxes': pred['boxes'][keep].cpu().numpy(),
            'masks': pred['masks'][keep].cpu().numpy(),
            'labels': pred['labels'][keep].cpu().numpy(),
            'scores': pred['scores'][keep].cpu().numpy()
        }

        return results, original_image



    def extract_edges_canny(self, mask, low_threshold=50, high_threshold=150):
        """使用Canny算子提取边缘"""
        # 将mask转换为uint8
        mask_uint8 = (mask * 255).astype(np.uint8)

        blurred = cv2.GaussianBlur(mask_uint8, (5, 5), 0)

        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        return edges

    def extract_edges_contour(self, mask):
        """使用轮廓检测提取边缘"""
        mask_uint8 = (mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        edges = np.zeros_like(mask_uint8)
        cv2.drawContours(edges, contours, -1, 255, 1)

        return edges, contours

    def extract_edges_morphology(self, mask):
        """使用形态学操作提取边缘"""
        mask_bool = mask > 0.5

        eroded = morphology.binary_erosion(mask_bool, morphology.disk(1))

        edges = mask_bool ^ eroded

        return edges.astype(np.uint8) * 255

    def refine_edges(self, edges):
        kernel = np.ones((3, 3), np.uint8)

        refined = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)

        return refined


def main():
    model_path = 'fault_detection_maskrcnn.pth'
    input_path = './data/severstal-steel-defect-detection/test_images'
    output_dir = './data/output'
    confidence_threshold = 0.3
    edge_method = 'canny'

    if os.path.exists(model_path):
        detector = DefectEdgeDetector(model_path)
        print("Using trained model")
    else:
        print(f"Model file {model_path} not found. Using pretrained model for demonstration.")
        detector = DefectEdgeDetector(use_pretrained=True)

    all_results = {}

    # 处理单张图像
    if os.path.isfile(input_path):
        print(f"Processing {input_path}...")
        try:
            image_name = os.path.splitext(os.path.basename(input_path))[0]
            results = detector.process_image(
                input_path,
                output_dir,
                confidence_threshold=confidence_threshold,
                edge_method=edge_method
            )
            all_results[image_name] = results

            if results and len(results['masks']) > 0:
                print(f"Found {len(results['masks'])} defects")
            else:
                print("No defects detected")
        except Exception as e:
            print(f"Error processing image: {e}")

    # 批量处理目录中的图像
    elif os.path.isdir(input_path):
        print(f"Processing images in {input_path}...")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        for filename in os.listdir(input_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(input_path, filename)
                image_name = os.path.splitext(filename)[0]
                print(f"Processing {filename}...")

                try:
                    results = detector.process_image(
                        image_path,
                        output_dir,
                        confidence_threshold=confidence_threshold,
                        edge_method=edge_method
                    )
                    all_results[image_name] = results

                    if results and len(results['masks']) > 0:
                        print(f"  Found {len(results['masks'])} defects")
                    else:
                        print(f"  No defects detected")

                except Exception as e:
                    print(f"  Error processing {filename}: {e}")
                    all_results[image_name] = {'masks': [], 'labels': [], 'scores': [], 'boxes': []}

    else:
        print(f"Invalid input path: {input_path}")
        return

    if all_results:
        detector.save_results_to_csv(all_results, output_dir)
        print(f"\nProcessing complete. Results saved in {output_dir}")


if __name__ == '__main__':
    main()