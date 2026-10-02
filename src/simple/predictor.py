import os
import torch
import json
import logging
from transformers import AutoTokenizer
import pandas as pd
from typing import List, Dict, Any, Optional
import numpy as np

from src.simple.crf_layer import fix_bio_labels
from src.simple.model import BIOBERT_TAE_Transformer_CRF

logger = logging.getLogger(__name__)


class NERPredictor:
    def __init__(self, model_path, device=None):
        """
        NER模型预测器

        Args:
            model_path: 模型路径
            device: 计算设备
        """
        self.model_path = model_path

        # 加载配置
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 加载标签映射
        label_map_path = os.path.join(model_path, "label_map.json")
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)

        # 创建ID到标签的映射
        self.id_to_label = {int(idx): label for label, idx in self.label_map.items()}

        # 设置设备
        if device is None:
            self.device = torch.device(self.config["training"]["device"])
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["paths"]["biobert_path"])

        # 创建并加载模型
        self.model = self._load_model()

    def _load_model(self):
        """
        加载模型

        Returns:
            model: 加载好的模型
        """
        # 创建模型
        model = BIOBERT_TAE_Transformer_CRF(self.config, len(self.label_map))

        # 加载模型权重
        model_weights_path = os.path.join(self.model_path, "model.pt")
        model.load_state_dict(torch.load(model_weights_path, map_location=self.device))

        # 将模型移到设备上并设置为评估模式
        model.to(self.device)
        model.eval()

        logger.info(f"Model loaded from {self.model_path}")

        return model

    def tokenize_sentence(self, sentence: List[str]) -> Dict[str, Any]:
        """
        对句子进行tokenize

        Args:
            sentence: 词列表

        Returns:
            features: 特征字典
        """
        # BERT tokenization
        bert_tokens = ["[CLS]"]
        original_to_bert_map = {}

        for i, word in enumerate(sentence):
            original_to_bert_map[i] = len(bert_tokens)
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = ["[UNK]"]
            bert_tokens.extend(word_tokens)

        bert_tokens.append("[SEP]")

        # 转换为输入IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        # 创建attention mask
        attention_mask = [1] * len(input_ids)

        # 填充序列至max_seq_length
        max_seq_length = self.config["model"]["max_seq_length"]
        padding_length = max_seq_length - len(input_ids)

        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
        elif padding_length < 0:
            # 序列过长，进行截断
            input_ids = input_ids[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
            bert_tokens = bert_tokens[:max_seq_length]

        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            "bert_tokens": bert_tokens,
            "original_to_bert_map": original_to_bert_map
        }

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """
        预测句子的NER标签

        Args:
            sentences: 句子列表，每个句子是一个词列表

        Returns:
            predictions: 预测标签列表
        """
        all_predictions = []

        # 禁用梯度计算
        with torch.no_grad():
            for sentence in sentences:
                # Tokenize句子
                features = self.tokenize_sentence(sentence)

                # 将特征移到设备上
                input_ids = features["input_ids"].to(self.device)
                attention_mask = features["attention_mask"].to(self.device)

                # 获取预测
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # 获取预测结果
                predictions = outputs["predictions"]

                # 将预测ID转换为标签
                pred_labels = []
                original_to_bert_map = features["original_to_bert_map"]

                for i, word in enumerate(sentence):
                    bert_idx = original_to_bert_map[i]

                    # 确保索引在范围内
                    if bert_idx < len(predictions[0]):
                        pred_id = predictions[0][bert_idx]
                        pred_label = self.id_to_label.get(pred_id, "O")
                    else:
                        pred_label = "O"

                    pred_labels.append(pred_label)
                pred_labels = fix_bio_labels(pred_labels)
                all_predictions.append(pred_labels)

        return all_predictions

    def predict_conll_file(self, input_file: str, output_file: str) -> None:
        """
        预测CONLL格式文件中的NER标签并保存结果

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        # 读取输入文件
        sentences = []
        current_sentence = []

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    parts = line.split()
                    if len(parts) >= 1:
                        token = parts[0]
                        current_sentence.append(token)

        # 处理最后一个句子
        if current_sentence:
            sentences.append(current_sentence)

        # 预测标签
        predictions = self.predict(sentences)

        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (sentence, sent_preds) in enumerate(zip(sentences, predictions)):
                for j, (token, pred) in enumerate(zip(sentence, sent_preds)):
                    f.write(f"{token} {pred}\n")

                # 句子之间添加空行
                if i < len(sentences) - 1:
                    f.write("\n")

        logger.info(f"Predictions saved to {output_file}")

    def evaluate_conll_file(self, input_file: str, output_file: str = None) -> Dict[str, float]:
        """
        评估CONLL格式文件中的NER标签

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径（可选）

        Returns:
            metrics: 评估指标
        """
        # 读取输入文件
        sentences = []
        gold_labels = []
        current_sentence = []
        current_labels = []

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        gold_labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[1]
                        current_sentence.append(token)
                        current_labels.append(label)

        # 处理最后一个句子
        if current_sentence:
            sentences.append(current_sentence)
            gold_labels.append(current_labels)

        # 预测标签
        pred_labels = self.predict(sentences)

        # 计算评估指标
        from utils.metrics_new import compute_metrics, format_metrics, save_metrics_to_file

        # 将标签展平为一维列表
        flat_gold = [label for sent_labels in gold_labels for label in sent_labels]
        flat_pred = [label for sent_preds in pred_labels for label in sent_preds]

        # 计算指标
        label_to_id = {label: idx for idx, label in self.id_to_label.items()}
        gold_ids = [label_to_id.get(label, label_to_id.get("O")) for label in flat_gold]
        pred_ids = [label_to_id.get(label, label_to_id.get("O")) for label in flat_pred]
        masks = [1] * len(flat_gold)

        metrics = compute_metrics(
            predictions=[pred_ids],
            label_ids=[gold_ids],
            id_to_label=self.id_to_label,
            attention_mask=[masks]
        )

        # 打印评估结果
        logger.info(format_metrics(metrics))

        # 如果提供了输出文件路径，写入预测结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, (sentence, gold, pred) in enumerate(zip(sentences, gold_labels, pred_labels)):
                    for j, (token, g, p) in enumerate(zip(sentence, gold, pred)):
                        f.write(f"{token} {g} {p}\n")

                    # 句子之间添加空行
                    if i < len(sentences) - 1:
                        f.write("\n")

            logger.info(f"Evaluation results saved to {output_file}")

        return metrics
