import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        """
        多头注意力机制

        Args:
            hidden_size: 隐藏层大小
            num_heads: 注意力头数
            dropout: dropout概率
        """
        super(MultiHeadAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        # 确保hidden_size可以被num_heads整除
        assert self.head_size * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        # 线性变换层
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, hidden_size)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        前向传播

        Args:
            query: 查询张量, shape: [batch_size, seq_length, hidden_size]
            key: 键张量, shape: [batch_size, seq_length, hidden_size]
            value: 值张量, shape: [batch_size, seq_length, hidden_size]
            mask: 掩码张量, shape: [batch_size, seq_length]

        Returns:
            outputs: 多头注意力的输出, shape: [batch_size, seq_length, hidden_size]
        """
        batch_size = query.size(0)

        # 线性变换
        q = self.q_linear(query)  # [batch_size, seq_length, hidden_size]
        k = self.k_linear(key)  # [batch_size, seq_length, hidden_size]
        v = self.v_linear(value)  # [batch_size, seq_length, hidden_size]

        # 调整形状以进行多头注意力计算
        q = q.view(batch_size, -1, self.num_heads, self.head_size).transpose(1,
                                                                             2)  # [batch_size, num_heads, seq_length, head_size]
        k = k.view(batch_size, -1, self.num_heads, self.head_size).transpose(1,
                                                                             2)  # [batch_size, num_heads, seq_length, head_size]
        v = v.view(batch_size, -1, self.num_heads, self.head_size).transpose(1,
                                                                             2)  # [batch_size, num_heads, seq_length, head_size]

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_size)  # [batch_size, num_heads, seq_length, seq_length]

        # 应用掩码（如果提供）
        if mask is not None:
            # 扩展mask以适应多头注意力
            extended_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_length]
            scores = scores.masked_fill(extended_mask == 0, -1e9)

        # 应用softmax获取注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_length, seq_length]
        attention_weights = self.dropout(attention_weights)

        # 计算加权和
        context = torch.matmul(attention_weights, v)  # [batch_size, num_heads, seq_length, head_size]

        # 重新调整形状
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.hidden_size)  # [batch_size, seq_length, hidden_size]

        # 最终线性变换
        output = self.output_linear(context)  # [batch_size, seq_length, hidden_size]

        return output


class TAEModule(nn.Module):
    def __init__(self, config):
        """
        任务感知的实体表示模块 (Task-Aware Entity Representation)

        Args:
            config: 配置信息
        """
        super(TAEModule, self).__init__()

        self.config = config
        self.hidden_size = config["model"]["tae"]["hidden_size"]
        self.num_heads = config["model"]["tae"]["num_heads"]

        # 多头注意力层
        self.self_attention = MultiHeadAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=config["model"]["tae"]["dropout"]
        )

        # 层归一化
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size)

        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config["model"]["tae"]["dropout"]),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(config["model"]["tae"]["dropout"])
        )

    def forward(self, sequence_output, attention_mask):
        """
        前向传播

        Args:
            sequence_output: BioBERT的输出特征, shape: [batch_size, seq_length, hidden_size]
            attention_mask: 注意力掩码, shape: [batch_size, seq_length]

        Returns:
            tae_output: TAE模块的输出特征, shape: [batch_size, seq_length, hidden_size]
        """
        # 残差连接和层归一化
        attention_output = self.self_attention(sequence_output, sequence_output, sequence_output, attention_mask)
        hidden_output = self.layer_norm1(sequence_output + attention_output)

        # 前馈神经网络
        ff_output = self.feed_forward(hidden_output)

        # 残差连接和层归一化
        tae_output = self.layer_norm2(hidden_output + ff_output)

        return tae_output
