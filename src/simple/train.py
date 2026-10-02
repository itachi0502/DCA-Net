import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
import json
from typing import Dict, List, Tuple, Any, Optional
import time
from datetime import datetime

from src.simple.model import BIOBERT_TAE_Transformer_CRF
from utils.metrics_new import compute_metrics, format_metrics, save_metrics_to_file

logger = logging.getLogger(__name__)


class NERTrainer:
    def __init__(self, config, train_loader, val_loader, test_loader, label_map, id_to_label):
        """
        NER模型训练器

        Args:
            config: 配置信息
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            label_map: 标签到ID的映射
            id_to_label: ID到标签的映射
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.label_map = label_map
        self.id_to_label = id_to_label
        self.num_labels = len(label_map)

        # 设置设备
        self.device = torch.device(config["training"]["device"])
        logger.info(f"Using device: {self.device}")

        # 创建模型
        self.model = BIOBERT_TAE_Transformer_CRF(config, self.num_labels, self.label_map, self.id_to_label)
        self.model.to(self.device)

        # 设置优化器和学习率调度器
        self.setup_optimizer_and_scheduler()

        # 创建输出目录
        self.create_output_dirs()

        # 初始化训练状态
        self.best_f1 = 0.0
        self.patience_counter = 0
        self.global_step = 0

    def setup_optimizer_and_scheduler(self):
        """
        设置优化器和学习率调度器
        """
        train_config = self.config["training"]

        # 将模型参数分组（不同的参数组可以有不同的学习率）
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': train_config["weight_decay"]},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        # 创建优化器
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=train_config["learning_rate"]
        )

        # 计算总的训练步数
        num_training_steps = len(self.train_loader) * train_config["num_epochs"]
        warmup_steps = int(num_training_steps * train_config["warmup_proportion"])

        # 创建学习率调度器
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda, -1)

        logger.info(f"Optimizer and scheduler setup complete")
        logger.info(f"Total training steps: {num_training_steps}, Warmup steps: {warmup_steps}")

    def create_output_dirs(self):
        """
        创建输出目录
        """
        # 创建输出目录
        self.output_dir = self.config["paths"]["output_dir"]
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 创建模型保存目录
        self.model_save_dir = self.config["paths"]["model_save_dir"]
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # 创建预测输出目录
        self.prediction_output_dir = self.config["paths"]["prediction_output"]
        if not os.path.exists(self.prediction_output_dir):
            os.makedirs(self.prediction_output_dir)

        logger.info(f"Created output directories")

    def train(self):
        """
        训练模型
        """
        train_config = self.config["training"]
        num_epochs = train_config["num_epochs"]

        logger.info(f"Starting training for {num_epochs} epochs")

        # 记录训练开始时间
        start_time = time.time()

        for epoch in range(num_epochs):
            logger.info(f"====== Epoch {epoch + 1}/{num_epochs} ======")

            # 训练一个epoch
            train_loss = self.train_epoch()
            logger.info(f"Training loss: {train_loss:.4f}")

            # 在验证集上评估
            val_metrics = self.evaluate(self.val_loader, "val")
            val_f1 = val_metrics["f1_micro"]

            logger.info(f"Validation F1 (micro): {val_f1:.4f}")

            # 检查是否达到最佳性能
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.patience_counter = 0

                # 保存最佳模型
                if train_config["save_best_model"]:
                    self.save_model("best_model")
                    logger.info(f"New best model saved with validation F1: {val_f1:.4f}")
            else:
                self.patience_counter += 1
                logger.info(
                    f"Validation F1 did not improve. Patience: {self.patience_counter}/{train_config['early_stopping_patience']}")

                # 检查是否应该早停
                if train_config["early_stopping_patience"] > 0 and self.patience_counter >= train_config[
                    "early_stopping_patience"]:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        # 计算训练时间
        training_time = time.time() - start_time
        logger.info(f"Training complete. Total time: {training_time:.2f} seconds")

        # 在测试集上进行最终评估
        logger.info("Evaluating on test set...")
        test_metrics = self.evaluate(self.test_loader, "test")

        # 输出最终结果
        logger.info("===== Final Test Results =====")
        logger.info(format_metrics(test_metrics))

        return test_metrics

    def train_epoch(self):
        """
        训练一个epoch

        Returns:
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = 0

        # 使用tqdm显示进度条
        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch in progress_bar:
            # 将批次数据移到设备上
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # 清除梯度
            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                label_ids=batch["label_ids"]
            )

            # 获取损失
            loss = outputs["loss"]

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["training"]["max_grad_norm"]
            )

            # 更新参数
            self.optimizer.step()
            self.scheduler.step()

            # 累计损失
            total_loss += loss.item()

            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # 更新全局步数
            self.global_step += 1

        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)

        return avg_loss

    def evaluate(self, data_loader, split="val"):
        """
        在数据集上评估模型

        Args:
            data_loader: 数据加载器
            split: 数据集划分名称

        Returns:
            metrics: 评估指标
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_masks = []

        # 禁用梯度计算
        with torch.no_grad():
            # 使用tqdm显示进度条
            progress_bar = tqdm(data_loader, desc=f"Evaluating on {split}")

            for batch in progress_bar:
                # 将批次数据移到设备上
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                # 前向传播
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )

                # 获取预测结果
                predictions = outputs["predictions"]
                labels = batch["label_ids"]
                masks = batch["attention_mask"]

                # 将预测结果添加到列表中
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
                all_masks.extend(masks.cpu().numpy())

        # 计算评估指标
        metrics = compute_metrics(
            predictions=all_predictions,
            label_ids=all_labels,
            id_to_label=self.id_to_label,
            attention_mask=all_masks
        )

        # 保存评估指标到文件
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        metrics_file = os.path.join(self.output_dir, f"{split}_metrics_{timestamp}.txt")
        save_metrics_to_file(metrics, metrics_file)

        return metrics

    def save_model(self, name="model"):
        """
        保存模型

        Args:
            name: 模型名称
        """
        # 创建保存路径
        save_path = os.path.join(self.model_save_dir, name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 保存模型状态
        model_path = os.path.join(save_path, "model.pt")
        torch.save(self.model.state_dict(), model_path)

        # 保存配置
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

        # 保存标签映射
        label_map_path = os.path.join(save_path, "label_map.json")
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.label_map, f, ensure_ascii=False, indent=2)

        logger.info(f"Model saved to {save_path}")

    def load_model(self, path):
        """
        加载模型

        Args:
            path: 模型路径
        """
        # 加载模型状态
        model_path = os.path.join(path, "model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        logger.info(f"Model loaded from {path}")

        return self.model
