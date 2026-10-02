import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class NERDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            tokenizer,
            max_seq_length: int = 128,
            label_map: Optional[Dict[str, int]] = None,
            is_training: bool = True
    ):
        """
        NER数据集加载类

        Args:
            data_path: CONLL格式的数据文件路径
            tokenizer: BERT tokenizer
            max_seq_length: 最大序列长度
            label_map: 标签到ID的映射字典
            is_training: 是否为训练模式
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_training = is_training

        # 读取数据
        self.sentences, self.labels = self._read_conll_file(data_path)

        # 如果没有提供label_map，则从训练数据中创建
        if label_map is None and is_training:
            self.label_map = self._create_label_map(self.labels)
        else:
            self.label_map = label_map

        if self.label_map is None:
            raise ValueError("No label map provided and not in training mode")

        # 确保特殊标签存在
        self._ensure_special_labels()

        self.id_to_label = {idx: label for label, idx in self.label_map.items()}
        logger.info(f"Created dataset with {len(self.sentences)} sentences")
        logger.info(f"Label map: {self.label_map}")

        # 将数据转换为特征
        self.features = self._convert_examples_to_features()

    def _ensure_special_labels(self):
        """
        确保特殊标签存在于标签映射中
        """
        special_labels = ["PAD", "[CLS]", "[SEP]", "O"]
        for label in special_labels:
            if label not in self.label_map:
                # 添加缺失的特殊标签
                self.label_map[label] = max(self.label_map.values()) + 1
                logger.warning(f"Special label '{label}' was not in label map. Added with id {self.label_map[label]}.")

    def _read_conll_file(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        """
        读取CONLL格式的文件，第一列为token，第二列为标签，中间用空格隔开，句子之间用空行分隔

        Args:
            file_path: 文件路径

        Returns:
            sentences: 句子列表，每个句子是一个token列表
            labels: 标签列表，每个句子对应一个标签列表
        """
        sentences = []
        labels = []

        current_sentence = []
        current_labels = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if not line:
                    # 空行表示句子结束
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                else:
                    # 拆分token和标签
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[1]
                        current_sentence.append(token)
                        current_labels.append(label)

        # 处理最后一个句子
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)

        return sentences, labels

    def _create_label_map(self, all_labels: List[List[str]]) -> Dict[str, int]:
        """
        从所有标签中创建标签映射字典

        Args:
            all_labels: 所有标签列表

        Returns:
            label_map: 标签到ID的映射字典
        """
        unique_labels = set()
        for labels in all_labels:
            unique_labels.update(labels)

        # 确保特殊标签在前面
        special_labels = ["O", "PAD", "[CLS]", "[SEP]"]
        ordered_labels = []

        # 首先添加特殊标签 - 即使它们不在原始数据中
        for label in special_labels:
            ordered_labels.append(label)
            if label in unique_labels:
                unique_labels.remove(label)

        # 然后添加其他标签（按字母顺序排序）
        ordered_labels.extend(sorted(list(unique_labels)))

        return {label: idx for idx, label in enumerate(ordered_labels)}

    def _convert_examples_to_features(self) -> List[Dict[str, Any]]:
        """
        将句子和标签转换为模型输入特征

        Returns:
            features: 特征列表
        """
        features = []

        for sentence_idx, (sentence, labels) in enumerate(zip(self.sentences, self.labels)):
            # BERT tokenization可能会将单词拆分成多个子词，需要对齐标签
            bert_tokens = []
            bert_labels = []
            bert_token_mapping = []  # 原始token到BERT子词的映射

            # 添加[CLS]标记
            bert_tokens.append("[CLS]")
            bert_labels.append("[CLS]")

            for word_idx, (word, label) in enumerate(zip(sentence, labels)):
                # 使用BERT tokenizer对单词进行分词
                word_tokens = self.tokenizer.tokenize(word)
                if not word_tokens:
                    # 如果tokenizer返回空列表，使用[UNK]
                    word_tokens = ["[UNK]"]

                # 添加分词后的tokens
                bert_tokens.extend(word_tokens)

                # 第一个子词使用原始标签，其他子词使用同样的标签
                bert_labels.extend([label] + ["PAD"] * (len(word_tokens) - 1))

                # 更新映射
                bert_token_mapping.extend([word_idx] * len(word_tokens))

            # 添加[SEP]标记
            bert_tokens.append("[SEP]")
            bert_labels.append("[SEP]")

            # 将tokens转换为ids
            input_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

            # 将标签转换为ids
            label_ids = [self.label_map.get(label, self.label_map["O"]) for label in bert_labels]

            # 创建attention mask
            attention_mask = [1] * len(input_ids)

            # 填充序列至max_seq_length
            padding_length = self.max_seq_length - len(input_ids)
            if padding_length > 0:
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
                label_ids += [self.label_map["PAD"]] * padding_length
            elif padding_length < 0:
                # 序列过长，进行截断
                input_ids = input_ids[:self.max_seq_length]
                attention_mask = attention_mask[:self.max_seq_length]
                label_ids = label_ids[:self.max_seq_length]

            # 确保所有序列都有正确的长度
            assert len(input_ids) == self.max_seq_length
            assert len(attention_mask) == self.max_seq_length
            assert len(label_ids) == self.max_seq_length

            features.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label_ids": label_ids,
                "sentence_idx": sentence_idx,
                "bert_tokens": bert_tokens[:self.max_seq_length],
                "original_tokens": sentence,
                "original_labels": labels
            })

        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]

        return {
            "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(feature["attention_mask"], dtype=torch.long),
            "label_ids": torch.tensor(feature["label_ids"], dtype=torch.long),
            "sentence_idx": feature["sentence_idx"]
        }


def get_data_loaders(
        config,
        tokenizer
):
    """
    获取训练、验证和测试数据加载器

    Args:
        config: 配置信息
        tokenizer: BERT tokenizer

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        label_map: 标签映射
    """
    # 创建训练数据集
    train_dataset = NERDataset(
        data_path=config["paths"]["train_data"],
        tokenizer=tokenizer,
        max_seq_length=config["model"]["max_seq_length"],
        is_training=True
    )

    # 使用训练数据集中的标签映射创建验证和测试数据集
    val_dataset = NERDataset(
        data_path=config["paths"]["val_data"],
        tokenizer=tokenizer,
        max_seq_length=config["model"]["max_seq_length"],
        label_map=train_dataset.label_map,
        is_training=False
    )

    test_dataset = NERDataset(
        data_path=config["paths"]["test_data"],
        tokenizer=tokenizer,
        max_seq_length=config["model"]["max_seq_length"],
        label_map=train_dataset.label_map,
        is_training=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, test_loader, train_dataset.label_map, train_dataset.id_to_label
