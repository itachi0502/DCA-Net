import torch
import torch.nn as nn
import transformers
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.bert.configuration_bert import BertConfig
import logging

logger = logging.getLogger(__name__)


class TransformerModule(nn.Module):
    def __init__(self, config):
        """
        Transformer编码器模块

        Args:
            config: 配置信息
        """
        super(TransformerModule, self).__init__()

        self.config = config
        self.hidden_size = config["model"]["transformer"]["hidden_size"]
        self.num_hidden_layers = config["model"]["transformer"]["num_hidden_layers"]

        # 创建标准的BertConfig对象，确保包含所有必要的参数
        bert_config = BertConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=config["model"]["transformer"]["num_attention_heads"],
            intermediate_size=config["model"]["transformer"]["intermediate_size"],
            hidden_dropout_prob=config["model"]["transformer"]["hidden_dropout_prob"],
            attention_probs_dropout_prob=config["model"]["transformer"]["attention_probs_dropout_prob"],
            max_position_embeddings=512,  # 默认值
            type_vocab_size=2,  # 默认值
            initializer_range=0.02,  # 默认值
            layer_norm_eps=1e-12,  # 添加缺失的layer_norm_eps参数
            pad_token_id=0,  # 默认值
            position_embedding_type="absolute",  # 默认值
            use_cache=True,  # 默认值
            classifier_dropout=None,  # 默认值
        )

        # 创建Transformer层堆栈
        self.layer = nn.ModuleList([
            BertLayer(bert_config) for _ in range(self.num_hidden_layers)
        ])

        # 层归一化
        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, hidden_states, attention_mask):
        """
        前向传播

        Args:
            hidden_states: 上一层的输出特征, shape: [batch_size, seq_length, hidden_size]
            attention_mask: 注意力掩码, shape: [batch_size, seq_length]

        Returns:
            transformer_output: Transformer模块的输出特征, shape: [batch_size, seq_length, hidden_size]
        """
        # 将attention_mask转换为transformers库所需的格式
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 通过Transformer层
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]

        # 应用层归一化
        transformer_output = self.layer_norm(hidden_states)

        return transformer_output
