import torch
import torch.nn as nn
import logging
from src.simple.biobert_encoder import BioBERTEncoder
from src.simple.tae_layer import TAEModule
from src.simple.transformer_module import TransformerModule
from src.simple.crf_layer import CRFModule

logger = logging.getLogger(__name__)


class BIOBERT_TAE_Transformer_CRF(nn.Module):
    def __init__(self, config, num_labels, label_map=None, id_to_label=None):
        """
        BIOBERT-TAE-Transformer-CRF模型

        Args:
            config: 配置信息
            num_labels: 标签数量
            label_map: 标签到ID的映射字典（可选）
            id_to_label: ID到标签的映射字典（可选）
        """
        super(BIOBERT_TAE_Transformer_CRF, self).__init__()

        self.config = config
        self.num_labels = num_labels
        self.label_map = label_map
        self.id_to_label = id_to_label

        # BioBERT编码器
        self.biobert_encoder = BioBERTEncoder(config)

        # 是否使用TAE模块
        self.use_tae = config["model"]["tae"]["use_tae"]
        if self.use_tae:
            self.tae_module = TAEModule(config)

        # Transformer模块
        self.transformer_module = TransformerModule(config)

        # CRF模块
        self.crf_module = CRFModule(config, num_labels, label_map, id_to_label)

    def forward(self, input_ids, attention_mask, label_ids=None):
        """
        前向传播

        Args:
            input_ids: 输入token ids, shape: [batch_size, seq_length]
            attention_mask: 注意力掩码, shape: [batch_size, seq_length]
            label_ids: 标签ids, shape: [batch_size, seq_length]

        Returns:
            outputs: 包含损失和/或预测的字典
        """
        # BioBERT编码
        sequence_output = self.biobert_encoder(input_ids, attention_mask)

        # TAE处理
        if self.use_tae:
            sequence_output = self.tae_module(sequence_output, attention_mask)

        # Transformer编码
        sequence_output = self.transformer_module(sequence_output, attention_mask)

        # CRF处理
        outputs = self.crf_module(sequence_output, label_ids, attention_mask)

        return outputs
