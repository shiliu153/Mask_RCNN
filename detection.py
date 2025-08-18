import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import get_model
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy import ndimage
import pandas as pd
import matplotlib.cm as cm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import multiprocessing as mp
from functools import partial


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

        # 启用模型优化
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 故障类别名称和颜色映射
        self.class_names = {1: 'fault_1', 2: 'fault_2', 3: 'fault_3', 4: 'fault_4'}
        self.class_colors = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255], 4: [255, 255, 0]}

        # 线程安全的缓存
        self._thread_local = threading.local()

        # GPU锁（如果使用GPU）
        self._gpu_lock = threading.Lock()

        # Matplotlib锁（用于图形生成）
        self._plot_lock = threading.Lock()

        # 缓存常用的内核
        self._cached_kernels = {
            'close': np.ones((3, 3), np.uint8),
            'dilate': np.ones((2, 2), np.uint8),
            'open': np.ones((2, 2), np.uint8)
        }

    def get_thread_local_arrays(self):
        """获取线程本地的临时数组"""
        if not hasattr(self._thread_local, 'temp_arrays'):
            self._thread_local.temp_arrays = {}
        return self._thread_local.temp_arrays

    def get_temp_array(self, shape, dtype=np.uint8):
        """线程安全的临时数组获取"""
        temp_arrays = self.get_thread_local_arrays()
        key = (shape, dtype)
        if key not in temp_arrays:
            temp_arrays[key] = np.zeros(shape, dtype=dtype)
        else:
            temp_arrays[key].fill(0)
        return temp_arrays[key]

    def process_single_image_worker(self, image_path, output_dir, confidence_threshold, edge_method):
        """单个图像处理工作函数（线程安全）"""
        try:
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            # 预处理图像
            image_tensor, original_image = self.preprocess_image(image_path)

            # GPU推理（需要锁保护）
            with self._gpu_lock:
                image_tensor = image_tensor.to(self.device)
                with torch.no_grad():
                    predictions = self.model(image_tensor)

                # 立即移到CPU释放GPU内存
                results = self.filter_predictions(predictions[0], confidence_threshold)

                # 获取原始结果
                raw_results = {
                    'boxes': predictions[0]['boxes'].cpu().numpy(),
                    'masks': predictions[0]['masks'].cpu().numpy(),
                    'labels': predictions[0]['labels'].cpu().numpy(),
                    'scores': predictions[0]['scores'].cpu().numpy()
                }

            # CPU密集型处理（不需要锁）
            self.process_single_image_complete_threadsafe(
                results, raw_results, original_image, image_name,
                output_dir, edge_method, image_path, confidence_threshold
            )

            return image_name, results, True

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return os.path.splitext(os.path.basename(image_path))[0], None, False

    def process_images_multithread(self, image_paths, output_dir, confidence_threshold=0.5,
                                   edge_method='canny', max_workers=None):
        """多线程并行处理图像"""
        if max_workers is None:
            # 根据GPU内存和CPU核心数自动设置
            if torch.cuda.is_available():
                max_workers = min(4, len(image_paths))  # GPU限制并发数
            else:
                max_workers = min(mp.cpu_count(), len(image_paths))

        print(f"Processing {len(image_paths)} images with {max_workers} threads...")

        all_results = {}
        successful_count = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(
                    self.process_single_image_worker,
                    image_path, output_dir, confidence_threshold, edge_method
                ): image_path
                for image_path in image_paths
            }

            for future in as_completed(future_to_path):
                image_path = future_to_path[future]
                try:
                    image_name, results, success = future.result()
                    if success:
                        all_results[image_name] = results
                        successful_count += 1
                        print(f"✓ Completed: {image_name}")
                    else:
                        failed_count += 1
                        print(f"✗ Failed: {image_name}")

                except Exception as e:
                    failed_count += 1
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    print(f"✗ Failed: {image_name} - {e}")

        print(f"\nProcessing summary: {successful_count} successful, {failed_count} failed")
        return all_results

    def process_single_image_complete_threadsafe(self, results, raw_results, original_image,
                                                 image_name, output_dir, edge_method, image_path, confidence_threshold):
        """线程安全的完整图像处理"""
        dirs_to_create = ['edges', 'heatmaps', 'analysis', 'overlays', 'pixel_scores', 'score_visualizations']
        for dir_name in dirs_to_create:
            dir_path = os.path.join(output_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)

        class_score_maps, combined_scores, combined_labels = self.save_pixel_scores_optimized(
            results, original_image.shape, image_name, output_dir
        )

        with self._plot_lock:
            self.create_score_visualization_optimized(
                class_score_maps, combined_scores, combined_labels, image_name, output_dir
            )

        # 处理边缘检测
        height, width = original_image.shape[:2]
        combined_edges = np.zeros((height, width), dtype=np.uint8)
        combined_masks = np.zeros((height, width), dtype=np.uint8)
        edge_results = []

        if results and len(results['masks']) > 0:
            for i, (mask, label, score, box) in enumerate(zip(
                    results['masks'], results['labels'], results['scores'], results['boxes']
            )):
                if mask.ndim == 3:
                    mask = mask[0]

                mask_resized = cv2.resize(mask, (width, height))
                mask_binary = (mask_resized > confidence_threshold).astype(np.uint8) * 255

                # 提取边缘
                edges = self.extract_edges_fast(mask_resized, edge_method)
                refined_edges = self.refine_edges(edges)

                # 合并到组合图像
                combined_masks = np.maximum(combined_masks, mask_binary)
                combined_edges = np.maximum(combined_edges, refined_edges)

                # 保存结果信息
                edge_results.append({
                    'mask_index': i,
                    'label': label,
                    'score': score,
                    'box': box,
                    'edge_pixels': np.sum(refined_edges > 0)
                })

        # 保存组合边缘
        edges_dir = os.path.join(output_dir, 'edges')
        cv2.imwrite(os.path.join(edges_dir, f"{image_name}_combined_edges.png"), combined_edges)

        # 创建和保存热力图
        heatmaps_dir = os.path.join(output_dir, 'heatmaps')

        if raw_results and 'masks' in raw_results and len(raw_results['masks']) > 0:
            raw_heatmap = self.create_raw_heatmap(raw_results, original_image.shape)
            raw_detections_count = len(raw_results['masks'])
        else:
            raw_heatmap = np.zeros_like(original_image)
            raw_detections_count = 0

        cv2.imwrite(os.path.join(heatmaps_dir, f"{image_name}_raw_heatmap.png"),
                    cv2.cvtColor(raw_heatmap, cv2.COLOR_RGB2BGR))

        filtered_heatmap = self.create_heatmap(results, original_image.shape)
        cv2.imwrite(os.path.join(heatmaps_dir, f"{image_name}_filtered_heatmap.png"),
                    cv2.cvtColor(filtered_heatmap, cv2.COLOR_RGB2BGR))

        # 创建分析图（使用线程锁）
        analysis_dir = os.path.join(output_dir, 'analysis')
        with self._plot_lock:
            self.create_analysis_plot(original_image, combined_masks, combined_edges,
                                      raw_heatmap, edge_results, image_name, analysis_dir,
                                      edge_method, raw_detections_count)

        # 创建叠加图
        overlays_dir = os.path.join(output_dir, 'overlays')
        overlay_raw = cv2.addWeighted(original_image, 0.6, raw_heatmap, 0.4, 0)
        cv2.imwrite(os.path.join(overlays_dir, f"{image_name}_overlay_raw_heatmap.png"),
                    cv2.cvtColor(overlay_raw, cv2.COLOR_RGB2BGR))

        overlay_filtered = cv2.addWeighted(original_image, 0.6, filtered_heatmap, 0.4, 0)
        cv2.imwrite(os.path.join(overlays_dir, f"{image_name}_overlay_filtered_heatmap.png"),
                    cv2.cvtColor(overlay_filtered, cv2.COLOR_RGB2BGR))

    def create_score_visualization_optimized(self, class_score_maps, combined_scores,
                                             combined_labels, image_name, output_dir):
        """线程安全的分数可视化"""
        try:
            visualizations_dir = os.path.join(output_dir, 'score_visualizations')
            os.makedirs(visualizations_dir, exist_ok=True)

            # 关闭任何现有的图形
            plt.close('all')

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            # 类别分数图
            for i, (class_id, score_map) in enumerate(class_score_maps.items()):
                if i < 4:
                    im = axes[i].imshow(score_map, cmap='hot', vmin=0, vmax=1)
                    axes[i].set_title(f'Class {class_id} Scores')
                    axes[i].axis('off')
                    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

            # 组合分数图
            if len(class_score_maps) >= 1:
                im = axes[4].imshow(combined_scores, cmap='hot', vmin=0, vmax=1)
                axes[4].set_title('Combined Scores')
                axes[4].axis('off')
                plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

            # 预测类别图
            if len(class_score_maps) >= 1:
                label_colors = np.zeros((*combined_labels.shape, 3), dtype=np.uint8)
                for class_id in range(1, self.num_classes):
                    mask = combined_labels == class_id
                    color = self.class_colors.get(class_id, [128, 128, 128])
                    label_colors[mask] = color

                axes[5].imshow(label_colors)
                axes[5].set_title('Predicted Classes')
                axes[5].axis('off')

            # 隐藏未使用的子图
            for i in range(len(class_score_maps) + 2, 6):
                if i < 6:
                    axes[i].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(visualizations_dir, f"{image_name}_score_visualization.png"),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"Warning: Could not create score visualization for {image_name}: {e}")
            plt.close('all')

    def create_analysis_plot(self, original_image, combined_masks, combined_edges,
                             raw_heatmap, edge_results, image_name, analysis_dir,
                             edge_method, raw_detections_count):
        """线程安全的分析图创建"""
        try:
            # 关闭任何现有的图形
            plt.close('all')

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

            axes[3].imshow(raw_heatmap)
            axes[3].set_title('Raw Heatmap (All Detections)')
            axes[3].axis('off')

            # 添加注释
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

            # 统计信息
            edge_pixels = np.sum(combined_edges > 0)
            total_pixels = combined_edges.shape[0] * combined_edges.shape[1]
            edge_ratio = edge_pixels / total_pixels * 100

            stats_text = f"Edge pixels: {edge_pixels}\nEdge ratio: {edge_ratio:.2f}%\nFiltered defects: {len(edge_results)}\nRaw detections: {raw_detections_count}"
            axes[5].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
            axes[5].set_title('Statistics')
            axes[5].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"{image_name}_analysis.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"Warning: Could not create analysis plot for {image_name}: {e}")
            plt.close('all')

    # 保留所有其他原有方法...
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
        """优化的热力图创建"""
        height, width = original_shape[:2]
        heatmap = np.zeros((height, width, 3), dtype=np.float32)

        if not results or len(results['masks']) == 0:
            return (heatmap * 255).astype(np.uint8)

        for mask, label, score in zip(results['masks'], results['labels'], results['scores']):
            if mask.ndim == 3:
                mask = mask[0]

            mask_resized = cv2.resize(mask, (width, height))
            color = self.class_colors.get(label, [128, 128, 128])

            # 向量化操作
            color_array = np.array(color) / 255.0
            weighted_mask = mask_resized * score

            for c in range(3):
                channel_value = weighted_mask * color_array[c]
                heatmap[:, :, c] = np.maximum(heatmap[:, :, c], channel_value)

        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        return heatmap

    def save_pixel_scores_optimized(self, results, original_shape, image_name, output_dir):
        """优化的像素分数保存"""
        height, width = original_shape[:2]
        pixel_scores_dir = os.path.join(output_dir, 'pixel_scores')
        os.makedirs(pixel_scores_dir, exist_ok=True)

        # 预分配数组
        class_score_maps = {}
        for class_id in range(1, self.num_classes):
            class_score_maps[class_id] = np.zeros((height, width), dtype=np.float32)

        if results and len(results['masks']) > 0:
            # 向量化处理
            for mask, label, score in zip(results['masks'], results['labels'], results['scores']):
                if mask.ndim == 3:
                    mask = mask[0]

                mask_resized = cv2.resize(mask, (width, height))

                if label in class_score_maps:
                    weighted_score = mask_resized * score
                    class_score_maps[label] = np.maximum(class_score_maps[label], weighted_score)

        # 批量保存numpy文件
        for class_id, score_map in class_score_maps.items():
            npy_filename = f"{image_name}_class_{class_id}_scores.npy"
            np.save(os.path.join(pixel_scores_dir, npy_filename), score_map)

        # 计算组合分数
        combined_scores = np.zeros((height, width), dtype=np.float32)
        combined_labels = np.zeros((height, width), dtype=np.int32)

        for class_id, score_map in class_score_maps.items():
            better_mask = score_map > combined_scores
            combined_scores[better_mask] = score_map[better_mask]
            combined_labels[better_mask] = class_id

        # 保存组合结果
        np.save(os.path.join(pixel_scores_dir, f"{image_name}_combined_scores.npy"), combined_scores)
        np.save(os.path.join(pixel_scores_dir, f"{image_name}_combined_labels.npy"), combined_labels)

        # 只在图像较小时保存CSV（优化条件）
        if width * height < 1000000:
            self.save_pixel_csv(pixel_scores_dir, image_name, height, width,
                                combined_scores, combined_labels, class_score_maps)

        return class_score_maps, combined_scores, combined_labels

    def save_pixel_csv(self, pixel_scores_dir, image_name, height, width,
                       combined_scores, combined_labels, class_score_maps):
        """优化的CSV保存，使用向量化操作"""
        # 创建坐标网格
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        # 准备数据字典
        pixel_data = {
            'image_name': np.full(height * width, image_name),
            'x': x_coords.flatten(),
            'y': y_coords.flatten(),
            'combined_score': combined_scores.flatten(),
            'predicted_class': combined_labels.flatten()
        }

        # 添加类别分数
        for class_id in range(1, self.num_classes):
            pixel_data[f'class_{class_id}_score'] = class_score_maps[class_id].flatten()

        df_pixels = pd.DataFrame(pixel_data)
        csv_filename = f"{image_name}_pixel_scores.csv"
        df_pixels.to_csv(os.path.join(pixel_scores_dir, csv_filename), index=False)

    def extract_edges_fast(self, mask, method='canny'):
        """优化的快速边缘提取"""
        mask_normalized = mask if mask.max() <= 1.0 else mask / mask.max()
        mask_uint8 = (mask_normalized > 0.1).astype(np.uint8) * 255

        if mask_uint8.max() == 0:
            return np.zeros_like(mask_uint8)

        # 减少形态学操作
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE,
                                      self._cached_kernels['close'])

        if method == 'canny':
            # 减少高斯模糊
            blurred = cv2.GaussianBlur(mask_uint8, (3, 3), 0)
            edges = cv2.Canny(blurred, 20, 60)
        elif method == 'contour':
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            edges = np.zeros_like(mask_uint8)
            if contours:
                cv2.drawContours(edges, contours, -1, 255, 2)
        else:  # morphology
            # 简化形态学方法
            selem = np.ones((3, 3), dtype=bool)
            mask_bool = mask_normalized > 0.1
            eroded = ndimage.binary_erosion(mask_bool, selem)
            edges = (mask_bool ^ eroded).astype(np.uint8) * 255

        return edges

    def refine_edges(self, edges):
        """边缘细化"""
        if edges.max() == 0:
            return edges

        refined = cv2.morphologyEx(edges, cv2.MORPH_OPEN, self._cached_kernels['open'])
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, self._cached_kernels['close'])
        refined = cv2.dilate(refined, self._cached_kernels['dilate'], iterations=1)

        return refined

    def save_results_to_csv(self, all_results, output_dir):
        """线程安全的CSV结果保存"""
        csv_data = []

        for image_name, results in all_results.items():
            if results and len(results['masks']) > 0:
                for i, (mask, label, score, box) in enumerate(zip(
                        results['masks'], results['labels'], results['scores'], results['boxes']
                )):
                    rle = self.mask_to_rle(mask[0] if mask.ndim == 3 else mask)
                    csv_data.append({
                        'image_name': image_name,
                        'defect_id': i,
                        'class_id': label,
                        'class_name': self.class_names.get(label, f'fault_{label}'),
                        'confidence_score': score,
                        'bbox_x1': box[0],
                        'bbox_y1': box[1],
                        'bbox_x2': box[2],
                        'bbox_y2': box[3],
                        'rle_encoding': rle
                    })
            else:
                csv_data.append({
                    'image_name': image_name,
                    'defect_id': 0,
                    'class_id': 0,
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

    def preprocess_image(self, image_path):
        """优化的图像预处理"""
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor, original_image

    def filter_predictions(self, predictions, confidence_threshold):
        """从模型预测中过滤结果"""
        pred = predictions
        keep = pred['scores'] > confidence_threshold

        results = {
            'boxes': pred['boxes'][keep].cpu().numpy(),
            'masks': pred['masks'][keep].cpu().numpy(),
            'labels': pred['labels'][keep].cpu().numpy(),
            'scores': pred['scores'][keep].cpu().numpy()
        }
        return results

    def process_image(self, image_path, output_dir, confidence_threshold=0.5,
                      edge_method='canny', save_individual=True):
        """单个图像处理（保持向后兼容）"""
        image_name, results, success = self.process_single_image_worker(
            image_path, output_dir, confidence_threshold, edge_method
        )
        return results if success else None


def main():
    model_path = 'fault_detection_maskrcnn.pth'
    input_path = './data/severstal-steel-defect-detection/test_images'
    output_dir = './data/output'
    confidence_threshold = 0.8
    edge_method = 'contour'

    # 并行处理设置
    max_workers = 4  # 最大并行数

    # 创建检测器
    if os.path.exists(model_path):
        detector = DefectEdgeDetector(model_path)
        detector.model_path = model_path  # 保存路径用于多进程
        print("Using trained model")
    else:
        print(f"Model file {model_path} not found. Using pretrained model for demonstration.")
        detector = DefectEdgeDetector(use_pretrained=True)
        detector.model_path = None

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

    # 多线程处理目录中的图像
    elif os.path.isdir(input_path):
        print(f"Processing images in {input_path}...")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        # 收集所有图像路径
        image_paths = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if any(f.lower().endswith(ext) for ext in image_extensions)
        ]

        if not image_paths:
            print("No valid images found in the directory.")
            return

        print(f"Found {len(image_paths)} images")

        # 使用多线程处理
        all_results = detector.process_images_multithread(
            image_paths, output_dir, confidence_threshold,
            edge_method, max_workers
        )

    else:
        print(f"Invalid input path: {input_path}")
        return

    # 保存结果
    if all_results:
        detector.save_results_to_csv(all_results, output_dir)
        print(f"\nProcessing complete. Results saved in {output_dir}")
        print(f"Successfully processed {len(all_results)} images")


if __name__ == '__main__':
    main()