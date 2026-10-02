import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss实现，用于处理类别不平衡问题
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", ignore_index=0):
        """
        初始化Focal Loss

        Args:
            alpha: 平衡因子，用于调整正样本的权重
            gamma: 聚焦参数，降低易分类样本的权重
            reduction: 损失计算方式，可选 'none', 'mean', 'sum'
            ignore_index: 忽略的标签索引，通常是PAD标签
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, mask=None):
        """
        计算Focal Loss

        Args:
            inputs: 预测logits，shape [batch_size, seq_length, num_classes]
            targets: 目标标签，shape [batch_size, seq_length]
            mask: 掩码，shape [batch_size, seq_length]

        Returns:
            loss: 计算的损失值
        """
        batch_size, seq_length, num_classes = inputs.size()

        # 创建mask
        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool, device=inputs.device)
        else:
            mask = mask.bool()

        # 扁平化输入和目标
        inputs_flat = inputs.view(-1, num_classes)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        # 创建标签的one-hot编码
        one_hot = torch.zeros_like(inputs_flat)
        one_hot.scatter_(1, targets_flat.unsqueeze(1), 1)

        # 计算softmax概率
        probs = F.softmax(inputs_flat, dim=1)
        pt = torch.sum(one_hot * probs, dim=1)

        # 忽略指定的标签
        valid_mask = (targets_flat != self.ignore_index) & mask_flat

        # 计算Focal Loss权重
        focal_weight = (1 - pt) ** self.gamma

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(
            inputs_flat,
            targets_flat,
            reduction="none",
            ignore_index=self.ignore_index
        )

        # 应用Focal Loss权重
        focal_loss = focal_weight * ce_loss

        # 应用mask
        focal_loss = focal_loss * valid_mask.float()

        # 根据reduction方式计算结果
        if self.reduction == "mean":
            return focal_loss.sum() / (valid_mask.sum().float() + 1e-8)
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss.view(batch_size, seq_length)


def fix_bio_labels(label_sequence, label_to_id=None, id_to_label=None):
    """
    修复不符合BIO格式的标签序列

    Args:
        label_sequence: 标签序列列表或索引列表
        label_to_id: 标签到ID的映射字典（可选）
        id_to_label: ID到标签的映射字典（可选）

    Returns:
        fixed_labels: 修复后的标签序列
    """
    fixed_labels = []
    previous_type = None
    in_entity = False  # 跟踪是否在实体内部

    # 判断输入是标签字符串还是标签索引
    is_string_input = isinstance(label_sequence[0], str) if label_sequence else True

    for i, label in enumerate(label_sequence):
        # 处理字符串标签
        if is_string_input:
            # 获取当前标签类型和前缀
            if label.startswith('B-') or label.startswith('I-'):
                current_prefix = label[0]
                current_type = label[2:]
            else:
                current_prefix = label
                current_type = None

            # 应用修复规则
            if label.startswith('I-') and (previous_type is None or previous_type != current_type):
                # 规则1: I-标签前面没有对应的B-标签，将其转换为B-标签
                new_label = 'B-' + current_type
                in_entity = True
                previous_type = current_type
            elif label.startswith('B-') and in_entity and previous_type == current_type:
                # 规则2: 连续的B-标签，且类型相同，将后续的B-转为I-
                new_label = 'I-' + current_type
                previous_type = current_type
            else:
                # 其他情况保持不变
                new_label = label

                # 更新状态跟踪
                if label.startswith('B-'):
                    in_entity = True
                    previous_type = current_type
                elif label == 'O' or label in ['[CLS]', '[SEP]', 'PAD']:
                    in_entity = False
                    previous_type = None

            fixed_labels.append(new_label)

        # 处理标签索引
        else:
            if id_to_label is not None:
                label_str = id_to_label[label]

                # 获取当前标签类型和前缀
                if label_str.startswith('B-') or label_str.startswith('I-'):
                    current_prefix = label_str[0]
                    current_type = label_str[2:]
                else:
                    current_prefix = label_str
                    current_type = None

                # 应用修复规则
                if label_str.startswith('I-') and (previous_type is None or previous_type != current_type):
                    # 规则1: I-标签前面没有对应的B-标签
                    new_label_str = 'B-' + current_type
                    in_entity = True
                    previous_type = current_type

                    if label_to_id is not None and new_label_str in label_to_id:
                        fixed_labels.append(label_to_id[new_label_str])
                    else:
                        # 如果新标签不存在于映射中，保留原始标签
                        fixed_labels.append(label)
                elif label_str.startswith('B-') and in_entity and previous_type == current_type:
                    # 规则2: 连续的B-标签，且类型相同
                    new_label_str = 'I-' + current_type
                    previous_type = current_type

                    if label_to_id is not None and new_label_str in label_to_id:
                        fixed_labels.append(label_to_id[new_label_str])
                    else:
                        # 如果新标签不存在于映射中，保留原始标签
                        fixed_labels.append(label)
                else:
                    # 其他情况保持不变
                    fixed_labels.append(label)

                    # 更新状态跟踪
                    if label_str.startswith('B-'):
                        in_entity = True
                        previous_type = current_type
                    elif label_str == 'O' or label_str in ['[CLS]', '[SEP]', 'PAD']:
                        in_entity = False
                        previous_type = None
            else:
                fixed_labels.append(label)

    return fixed_labels


class CRF(nn.Module):
    """
    条件随机场模块
    """

    def __init__(self, num_tags, batch_first=True):
        """
        初始化CRF层

        Args:
            num_tags: 标签数量
            batch_first: 是否batch维度在第一位
        """
        super(CRF, self).__init__()

        self.num_tags = num_tags
        self.batch_first = batch_first

        # 转移矩阵参数
        # transitions[i, j]表示从标签j转移到标签i的概率
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # 初始化特殊的开始和结束转移
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """
        初始化CRF参数
        """
        # 使用均匀分布初始化
        nn.init.xavier_uniform_(self.transitions)
        nn.init.xavier_uniform_(self.start_transitions.unsqueeze(0))
        nn.init.xavier_uniform_(self.end_transitions.unsqueeze(0))

    def _compute_score(self, emissions, tags, mask):
        """
        计算给定标签序列的分数

        Args:
            emissions: 模型输出的发射分数, shape: [batch_size, seq_length, num_tags]
            tags: 真实标签序列, shape: [batch_size, seq_length]
            mask: 掩码, shape: [batch_size, seq_length]

        Returns:
            score: 分数
        """
        batch_size, seq_length = tags.shape
        mask = mask.float()

        # 开始转移分数
        score = self.start_transitions[tags[:, 0]]

        # 发射分数
        score += emissions[torch.arange(batch_size), 0, tags[:, 0]]

        for i in range(1, seq_length):
            # 转移分数
            score += self.transitions[tags[:, i], tags[:, i - 1]] * mask[:, i]

            # 发射分数
            score += emissions[torch.arange(batch_size), i, tags[:, i]] * mask[:, i]

        # 结束转移分数
        last_tag_indices = mask.sum(dim=1).long() - 1
        last_tags = tags[torch.arange(batch_size), last_tag_indices]

        # 添加到标签序列的结束转移分数
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions, mask):
        """
        计算配分函数Z(x)，即所有可能标签序列的分数和

        Args:
            emissions: 模型输出的发射分数, shape: [batch_size, seq_length, num_tags]
            mask: 掩码, shape: [batch_size, seq_length]

        Returns:
            normalizer: 配分函数值
        """
        batch_size, seq_length, _ = emissions.shape

        # 确保mask是布尔类型
        mask = mask.bool()

        # 初始化alpha为开始转移分数
        alpha = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_length):
            # 广播alpha以计算所有可能的转移
            broadcast_alpha = alpha.unsqueeze(2)  # [batch_size, num_tags, 1]

            # 广播发射分数
            broadcast_emissions = emissions[:, i].unsqueeze(1)  # [batch_size, 1, num_tags]

            # 计算下一步的分数: 前一步分数 + 转移分数 + 发射分数
            next_score = broadcast_alpha + self.transitions + broadcast_emissions

            # 对所有可能的来源标签求log-sum-exp
            next_score = torch.logsumexp(next_score, dim=1)

            # 使用mask更新alpha (确保使用布尔mask)
            mask_i = mask[:, i].unsqueeze(1)  # [batch_size, 1]
            alpha = torch.where(mask_i, next_score, alpha)

        # 添加结束转移分数
        alpha += self.end_transitions

        # 对所有可能的最终标签求log-sum-exp
        return torch.logsumexp(alpha, dim=1)

    def forward(self, emissions, tags, mask=None):
        """
        计算CRF的负对数似然损失

        Args:
            emissions: 模型输出的发射分数, shape: [batch_size, seq_length, num_tags]
            tags: 真实标签序列, shape: [batch_size, seq_length]
            mask: 掩码, shape: [batch_size, seq_length]

        Returns:
            loss: CRF负对数似然损失
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool, device=emissions.device)
        else:
            # 确保mask是布尔类型
            mask = mask.bool()

        # 确保mask和tags具有相同的形状
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape

        # 计算分子（给定标签序列的分数）
        numerator = self._compute_score(emissions, tags, mask)

        # 计算分母（配分函数）
        denominator = self._compute_normalizer(emissions, mask)

        # 计算负对数似然
        llh = numerator - denominator

        # 返回批量平均负对数似然
        return -llh.mean()

    def decode(self, emissions, mask=None):
        """
        使用Viterbi算法解码最佳标签序列

        Args:
            emissions: 模型输出的发射分数, shape: [batch_size, seq_length, num_tags]
            mask: 掩码, shape: [batch_size, seq_length]

        Returns:
            best_tags: 最佳标签序列列表
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        else:
            # 确保mask是布尔类型
            mask = mask.bool()

        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags

        batch_size, seq_length = mask.shape

        # 初始化Viterbi变量和回溯指针
        viterbi = self.start_transitions + emissions[:, 0]
        backpointers = torch.zeros(batch_size, seq_length, self.num_tags, dtype=torch.long, device=emissions.device)

        for i in range(1, seq_length):
            # 广播viterbi以计算所有可能的转移
            broadcast_viterbi = viterbi.unsqueeze(2)  # [batch_size, num_tags, 1]

            # 计算下一步的分数: 前一步分数 + 转移分数 + 发射分数
            next_score = broadcast_viterbi + self.transitions  # [batch_size, num_tags, num_tags]

            # 找到最佳前一个标签
            best_tags_and_score = next_score.max(dim=1)
            best_prev_tags = best_tags_and_score.indices  # [batch_size, num_tags]

            # 更新viterbi变量
            viterbi_tmp = best_tags_and_score.values + emissions[:, i]

            # 只在mask==True的位置更新
            mask_i = mask[:, i].unsqueeze(1)  # [batch_size, 1]
            viterbi = torch.where(mask_i, viterbi_tmp, viterbi)

            # 更新回溯指针
            backpointers[:, i] = best_prev_tags

        # 添加结束转移分数
        viterbi += self.end_transitions

        # 获取最佳结束标签和分数
        best_last_tag_indices = viterbi.argmax(dim=1)
        best_tags_list = []

        # 回溯找到最佳路径
        for idx in range(batch_size):
            best_tags = [best_last_tag_indices[idx].item()]

            # 计算有效长度
            valid_length = mask[idx].sum().item()

            # 回溯
            for i in range(valid_length - 1, 0, -1):
                best_tags.append(backpointers[idx, i, best_tags[-1]].item())

            # 反转标签列表，使其从开始到结束
            best_tags.reverse()

            # 添加到结果列表
            best_tags_list.append(best_tags)

        return best_tags_list


class CRFModule(nn.Module):
    def __init__(self, config, num_tags, label_map=None, id_to_label=None):
        """
        CRF模块

        Args:
            config: 配置信息
            num_tags: 标签数量
            label_map: 标签到ID的映射字典（可选）
            id_to_label: ID到标签的映射字典（可选）
        """
        super(CRFModule, self).__init__()

        self.config = config
        self.hidden_size = config["model"]["transformer"]["hidden_size"]
        self.num_tags = num_tags

        # 保存标签映射
        self.label_map = label_map
        self.id_to_label = id_to_label

        # 线性分类层，将隐藏状态映射到标签空间
        self.classifier = nn.Linear(self.hidden_size, self.num_tags)

        # CRF层
        self.use_crf = config["model"]["crf"]["use_crf"]
        if self.use_crf:
            self.crf = CRF(num_tags=self.num_tags)

        # 检查是否需要修复BIO标签
        self.fix_bio_tags = config["model"].get("fix_bio_tags", False)

        # 初始化Focal Loss (如果配置中包含)
        use_focal_loss = config["model"].get("loss_type", "default") == "focal_loss"
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            focal_params = config["model"].get("focal_loss", {})
            self.focal_loss = FocalLoss(
                alpha=focal_params.get("alpha", 0.25),
                gamma=focal_params.get("gamma", 2.0),
                reduction=focal_params.get("reduction", "mean"),
                ignore_index=0  # PAD标签
            )
            # 获取Focal Loss权重 (用于结合CRF和Focal Loss)
            self.focal_weight = focal_params.get("weight", 0.5)
        else:
            self.focal_loss = None
            self.focal_weight = 0.0

    def forward(self, sequence_output, labels=None, attention_mask=None):
        """
        前向传播

        Args:
            sequence_output: Transformer的输出特征, shape: [batch_size, seq_length, hidden_size]
            labels: 真实标签, shape: [batch_size, seq_length]
            attention_mask: 注意力掩码, shape: [batch_size, seq_length]

        Returns:
            outputs: 字典，包含损失和/或预测
        """
        # 将特征映射到标签空间
        logits = self.classifier(sequence_output)  # [batch_size, seq_length, num_tags]

        outputs = {}

        # 处理不同的损失函数情况
        if labels is not None:
            # 计算损失
            if self.use_crf:
                # CRF损失
                crf_loss = self.crf(logits, labels, attention_mask)

                if self.use_focal_loss:
                    # 结合Focal Loss和CRF损失
                    focal_loss = self.focal_loss(logits, labels, attention_mask)
                    # 加权组合
                    loss = (1 - self.focal_weight) * crf_loss + self.focal_weight * focal_loss
                    outputs["crf_loss"] = crf_loss.item()
                    outputs["focal_loss"] = focal_loss.item()
                else:
                    # 只使用CRF损失
                    loss = crf_loss

                outputs["loss"] = loss
            else:
                # 不使用CRF的情况
                if self.use_focal_loss:
                    # 使用Focal Loss
                    loss = self.focal_loss(logits, labels, attention_mask)
                else:
                    # 使用标准交叉熵损失
                    loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # 假设0是PAD标签
                    active_loss = attention_mask.bool().view(-1)
                    active_logits = logits.view(-1, self.num_tags)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)

                outputs["loss"] = loss

        # 获取预测结果
        if self.use_crf:
            # Viterbi解码
            best_tags = self.crf.decode(logits, attention_mask)

            # 如果需要修复BIO标签且提供了标签映射
            if self.fix_bio_tags and self.id_to_label is not None and self.label_map is not None:
                fixed_best_tags = []
                for tags in best_tags:
                    # 修复标签序列
                    fixed_tags = fix_bio_labels(tags, self.label_map, self.id_to_label)
                    fixed_best_tags.append(fixed_tags)
                outputs["predictions"] = fixed_best_tags
            else:
                outputs["predictions"] = best_tags
        else:
            # 简单的argmax预测
            predictions = torch.argmax(logits, dim=-1)

            # 如果需要修复BIO标签且提供了标签映射
            if self.fix_bio_tags and self.id_to_label is not None and self.label_map is not None:
                # 转换为CPU并转为列表
                predictions_list = predictions.cpu().tolist()
                fixed_predictions = []

                for pred_seq in predictions_list:
                    # 修复标签序列
                    fixed_pred_seq = fix_bio_labels(pred_seq, self.label_map, self.id_to_label)
                    fixed_predictions.append(fixed_pred_seq)

                # 转回张量
                fixed_predictions = torch.tensor(
                    fixed_predictions,
                    device=predictions.device,
                    dtype=predictions.dtype
                )
                outputs["predictions"] = fixed_predictions
            else:
                outputs["predictions"] = predictions

        return outputs
