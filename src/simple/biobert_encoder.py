import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import logging

logger = logging.getLogger(__name__)


class BioBERTEncoder(nn.Module):
    def __init__(self, config):
        """
        BioBERT编码器

        Args:
            config: 配置信息
        """
        super(BioBERTEncoder, self).__init__()

        self.config = config
        self.bert_config = AutoConfig.from_pretrained(config["paths"]["biobert_path"])

        # 加载BioBERT预训练模型
        logger.info(f"Loading BioBERT model from {config['paths']['biobert_path']}")
        self.biobert = AutoModel.from_pretrained(
            config["paths"]["biobert_path"],
            config=self.bert_config
        )

        # 设置隐藏层大小
        self.hidden_size = self.bert_config.hidden_size

        # 添加dropout层
        self.dropout = nn.Dropout(config["model"]["dropout"])

    def forward(self, input_ids, attention_mask):
        """
        前向传播

        Args:
            input_ids: 输入的token ids, shape: [batch_size, seq_length]
            attention_mask: 注意力掩码, shape: [batch_size, seq_length]

        Returns:
            outputs: BioBERT的输出特征, shape: [batch_size, seq_length, hidden_size]
        """
        # 获取BioBERT的输出
        outputs = self.biobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 获取序列输出
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]

        # 应用dropout
        sequence_output = self.dropout(sequence_output)

        return sequence_output
