import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import get_model
import os
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy import ndimage


class DefectEdgeDetector:
    def __init__(self, model_path=None, num_classes=5, device=None, use_pretrained=False):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes

        # 加载模型
        self.model = get_model(num_classes)

        if model_path and os.path.exists(model_path):
            # 加载训练好的模型
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded trained model from {model_path}")
        elif use_pretrained:
            # 使用预训练模型进行演示
            print("Using pretrained model for demonstration")
        else:
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Set use_pretrained=True to use pretrained model for testing.")

        self.model.to(self.device)
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 故障类别名称
        self.class_names = {1: 'fault_1', 2: 'fault_2', 3: 'fault_3', 4: 'fault_4'}

    def preprocess_image(self, image_path):
        """预处理输入图像"""
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)

        # 转换为tensor
        image_tensor = self.transform(image).unsqueeze(0)

        return image_tensor, original_image

    def detect_defects(self, image_path, confidence_threshold=0.5):
        """检测图像中的缺陷"""
        image_tensor, original_image = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        # 获取预测结果
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

        # 应用高斯模糊
        blurred = cv2.GaussianBlur(mask_uint8, (5, 5), 0)

        # Canny边缘检测
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        return edges

    def extract_edges_contour(self, mask):
        """使用轮廓检测提取边缘"""
        # 将mask转换为uint8
        mask_uint8 = (mask * 255).astype(np.uint8)

        # 查找轮廓
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建边缘图像
        edges = np.zeros_like(mask_uint8)
        cv2.drawContours(edges, contours, -1, 255, 1)

        return edges, contours

    def extract_edges_morphology(self, mask):
        """使用形态学操作提取边缘"""
        # 将mask转换为布尔类型
        mask_bool = mask > 0.5

        # 腐蚀操作
        eroded = morphology.binary_erosion(mask_bool, morphology.disk(1))

        # 边缘 = 原始mask - 腐蚀后的mask
        edges = mask_bool ^ eroded

        return edges.astype(np.uint8) * 255

    def refine_edges(self, edges):
        """细化边缘"""
        # 形态学操作去除噪声
        kernel = np.ones((3, 3), np.uint8)

        # 开运算去除小的噪声点
        refined = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

        # 闭运算连接断开的边缘
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)

        return refined

    def process_image(self, image_path, output_dir, confidence_threshold=0.5,
                      edge_method='canny', save_individual=True):
        """处理单张图像的完整流程"""
        # 检测缺陷
        results, original_image = self.detect_defects(image_path, confidence_threshold)

        if len(results['masks']) == 0:
            print(f"No defects detected in {image_path}")
            return None

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取图像文件名
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # 创建可视化图像
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # 显示原始图像
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 创建合并的边缘图像
        height, width = original_image.shape[:2]
        combined_edges = np.zeros((height, width), dtype=np.uint8)
        combined_masks = np.zeros((height, width), dtype=np.uint8)

        edge_results = []

        # 处理每个检测到的缺陷
        for i, (mask, label, score, box) in enumerate(zip(
                results['masks'], results['labels'], results['scores'], results['boxes']
        )):
            # 获取mask (移除批次维度)
            if mask.ndim == 3:
                mask = mask[0]

            # 调整mask大小到原图尺寸
            mask_resized = cv2.resize(mask, (width, height))

            # 根据选择的方法提取边缘
            if edge_method == 'canny':
                edges = self.extract_edges_canny(mask_resized)
            elif edge_method == 'contour':
                edges, contours = self.extract_edges_contour(mask_resized)
            elif edge_method == 'morphology':
                edges = self.extract_edges_morphology(mask_resized)
            else:
                raise ValueError("edge_method must be 'canny', 'contour', or 'morphology'")

            # 细化边缘
            edges_refined = self.refine_edges(edges)

            # 添加到合并图像
            combined_edges = cv2.bitwise_or(combined_edges, edges_refined)
            combined_masks = cv2.bitwise_or(combined_masks, (mask_resized * 255).astype(np.uint8))

            # 保存单个缺陷的边缘
            if save_individual:
                fault_name = self.class_names.get(label, f'fault_{label}')
                edge_filename = f"{image_name}_{fault_name}_{i}_edges.png"
                cv2.imwrite(os.path.join(output_dir, edge_filename), edges_refined)

            edge_results.append({
                'label': label,
                'score': score,
                'box': box,
                'edges': edges_refined,
                'mask': mask_resized
            })

        # 显示检测结果
        axes[1].imshow(combined_masks, cmap='gray')
        axes[1].set_title('Detected Masks')
        axes[1].axis('off')

        # 显示边缘检测结果
        axes[2].imshow(combined_edges, cmap='gray')
        axes[2].set_title(f'Edges ({edge_method})')
        axes[2].axis('off')

        # 显示叠加图像
        overlay = original_image.copy()
        overlay_colored = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        # 将边缘叠加到原图上
        edge_colored = cv2.applyColorMap(combined_edges, cv2.COLORMAP_JET)
        overlay_result = cv2.addWeighted(overlay_colored, 0.7, edge_colored, 0.3, 0)
        overlay_result = cv2.cvtColor(overlay_result, cv2.COLOR_BGR2RGB)

        axes[3].imshow(overlay_result)
        axes[3].set_title('Overlay Result')
        axes[3].axis('off')

        # 显示带标注的检测框
        annotated = original_image.copy()
        for result in edge_results:
            box = result['box'].astype(int)
            label = result['label']
            score = result['score']
            fault_name = self.class_names.get(label, f'fault_{label}')

            # 绘制检测框
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

            # 添加标签
            label_text = f"{fault_name}: {score:.2f}"
            cv2.putText(annotated, label_text, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        axes[4].imshow(annotated)
        axes[4].set_title('Detection Results')
        axes[4].axis('off')

        # 边缘统计信息
        edge_pixels = np.sum(combined_edges > 0)
        total_pixels = combined_edges.shape[0] * combined_edges.shape[1]
        edge_ratio = edge_pixels / total_pixels * 100

        stats_text = f"Edge pixels: {edge_pixels}\nEdge ratio: {edge_ratio:.2f}%\nDefects found: {len(edge_results)}"
        axes[5].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        axes[5].set_title('Statistics')
        axes[5].axis('off')

        plt.tight_layout()

        # 保存可视化结果
        plt.savefig(os.path.join(output_dir, f"{image_name}_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 保存合并的边缘图像
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_combined_edges.png"), combined_edges)
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_overlay.png"),
                    cv2.cvtColor(overlay_result, cv2.COLOR_RGB2BGR))

        return edge_results


def main():
    model_path = 'fault_detection_maskrcnn.pth'
    input_image = './data/train/train_images/0a1cade03.jpg'  # 输入图像路径
    output_dir = 'edge_detection_results'
    confidence_threshold = 0.5
    edge_method = 'canny'  # 可选: 'canny', 'contour', 'morphology'

    # 检查输入图片是否存在
    if not os.path.exists(input_image):
        print(f"Input image not found: {input_image}")
        print("Please check the image path.")
        return

    # 创建边缘检测器 - 只创建一次
    if os.path.exists(model_path):
        detector = DefectEdgeDetector(model_path)
        print("Using trained model")
    else:
        print(f"Model file {model_path} not found. Using pretrained model for demonstration.")
        detector = DefectEdgeDetector(use_pretrained=True)

    # 处理单张图像
    if os.path.isfile(input_image):
        print(f"Processing {input_image}...")
        try:
            results = detector.process_image(
                input_image,
                output_dir,
                confidence_threshold=confidence_threshold,
                edge_method=edge_method
            )

            if results:
                print(f"Found {len(results)} defects")
                for i, result in enumerate(results):
                    label = result['label']
                    score = result['score']
                    fault_name = detector.class_names.get(label, f'fault_{label}')
                    print(f"Defect {i + 1}: {fault_name} (confidence: {score:.3f})")
            else:
                print("No defects detected")
        except Exception as e:
            print(f"Error processing image: {e}")

    # 批量处理目录中的图像
    elif os.path.isdir(input_image):
        print(f"Processing images in {input_image}...")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        for filename in os.listdir(input_image):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(input_image, filename)
                print(f"Processing {filename}...")

                try:
                    results = detector.process_image(
                        image_path,
                        output_dir,
                        confidence_threshold=confidence_threshold,
                        edge_method=edge_method
                    )

                    if results:
                        print(f"  Found {len(results)} defects")
                    else:
                        print(f"  No defects detected")

                except Exception as e:
                    print(f"  Error processing {filename}: {e}")

    else:
        print(f"Invalid input path: {input_image}")


if __name__ == '__main__':
    main()