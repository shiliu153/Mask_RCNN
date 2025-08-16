import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import get_model
from dataset import FaultDetectionDataset, get_transform
import os
from tqdm import tqdm


def collate_fn(batch):
    """自定义批处理函数"""
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0

    for images, targets in tqdm(data_loader, desc=f'Epoch {epoch}'):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 前向传播
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)


def main():
    # 配置参数
    num_classes = 5  # 4种故障 + 背景
    batch_size = 2  # 根据GPU内存调整
    num_epochs = 100
    learning_rate = 0.005

    # 数据路径
    train_csv = './data/severstal-steel-defect-detection/train.csv'
    train_img_dir = './data/train/train_images'

    # 设备配置
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    # 创建数据集和数据加载器
    train_dataset = FaultDetectionDataset(
        csv_file=train_csv,
        img_dir=train_img_dir,
        transforms=get_transform(train=True)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    # 创建模型
    model = get_model(num_classes)
    model.to(device)

    # 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005
    )

    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.1
    )

    # 训练循环
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()

        print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoint_epoch_{epoch + 1}.pth')

    # 保存最终模型
    torch.save(model.state_dict(), 'fault_detection_maskrcnn.pth')
    print('Training completed!')


if __name__ == '__main__':
    main()