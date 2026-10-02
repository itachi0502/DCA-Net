import numpy as np
from typing import List, Dict, Union, Tuple
from sklearn.metrics import precision_recall_fscore_support, classification_report
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def convert_predictions_to_labels(
        predictions: List[List[int]],
        label_ids: List[List[int]],
        id_to_label: Dict[int, str],
        attention_mask: List[List[int]]
) -> Tuple[List[str], List[str]]:
    """
    将预测的ID转换为标签字符串

    Args:
        predictions: 模型预测的标签ID
        label_ids: 真实标签ID
        id_to_label: ID到标签的映射字典
        attention_mask: 注意力掩码

    Returns:
        true_labels: 真实标签字符串列表
        pred_labels: 预测标签字符串列表
    """
    true_labels = []
    pred_labels = []

    # 定义要排除的特殊标签
    special_labels = ["PAD", "[CLS]", "[SEP]"]

    for pred_seq, true_seq, mask in zip(predictions, label_ids, attention_mask):
        # 如果预测是列表（CRF返回列表），则转换为张量
        if isinstance(pred_seq, list):
            # 将pred_seq转换为与true_seq相同长度的列表，用PAD填充
            if len(pred_seq) < len(true_seq):
                # CRF返回的长度可能不包含PAD，需要填充
                pred_seq = pred_seq + [0] * (len(true_seq) - len(pred_seq))
            elif len(pred_seq) > len(true_seq):
                # 如果预测太长，截断
                pred_seq = pred_seq[:len(true_seq)]

        # 根据mask提取非PAD的标签
        for i, m in enumerate(mask):
            if m == 1 and i > 0:  # 忽略[CLS]
                true_label = id_to_label.get(true_seq[i], "O")

                # 获取预测标签，确保索引有效
                if i < len(pred_seq):
                    if isinstance(pred_seq, list):
                        pred_label = id_to_label.get(pred_seq[i], "O")
                    else:
                        pred_label = id_to_label.get(pred_seq[i].item(), "O")
                else:
                    pred_label = "O"

                # 跳过特殊标签
                if true_label not in special_labels:
                    true_labels.append(true_label)
                    pred_labels.append(pred_label)

    return true_labels, pred_labels


def compute_metrics(
        predictions: Union[List[List[int]], np.ndarray],
        label_ids: Union[List[List[int]], np.ndarray],
        id_to_label: Dict[int, str],
        attention_mask: Union[List[List[int]], np.ndarray]
) -> Dict[str, float]:
    """
    计算NER评估指标

    Args:
        predictions: 模型预测的标签ID
        label_ids: 真实标签ID
        id_to_label: ID到标签的映射字典
        attention_mask: 注意力掩码

    Returns:
        metrics: 包含评估指标的字典
    """
    # 将预测和真实标签转换为列表格式
    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
    if isinstance(label_ids, np.ndarray):
        label_ids = label_ids.tolist()
    if isinstance(attention_mask, np.ndarray):
        attention_mask = attention_mask.tolist()

    # 转换为标签字符串
    true_labels, pred_labels = convert_predictions_to_labels(
        predictions, label_ids, id_to_label, attention_mask
    )

    # 计算精确率、召回率和F1分数（宏观和微观）
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='micro', zero_division=0
    )

    # 按每个标签计算指标，从报告中排除特殊标签
    labels = sorted(set(true_labels + pred_labels))
    # 过滤掉PAD标签
    labels = [label for label in labels if label not in ["PAD", "[CLS]", "[SEP]"]]

    report = classification_report(
        true_labels, pred_labels, labels=labels, output_dict=True, zero_division=0
    )

    # 创建结果字典
    metrics = {
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "detailed_report": report
    }

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    格式化评估指标为易读的字符串

    Args:
        metrics: 评估指标字典

    Returns:
        formatted_metrics: 格式化的指标字符串
    """
    lines = [
        "===== Evaluation Metrics =====",
        f"Macro Precision: {metrics['precision_macro']:.4f}",
        f"Macro Recall: {metrics['recall_macro']:.4f}",
        f"Macro F1: {metrics['f1_macro']:.4f}",
        f"Micro Precision: {metrics['precision_micro']:.4f}",
        f"Micro Recall: {metrics['recall_micro']:.4f}",
        f"Micro F1: {metrics['f1_micro']:.4f}",
        "\n===== Detailed Report ====="
    ]

    # 添加详细报告
    if "detailed_report" in metrics:
        report = metrics["detailed_report"]
        df = pd.DataFrame(report).T

        # 过滤掉不想显示的标签
        if "PAD" in df.index:
            df = df.drop("PAD")
        if "[CLS]" in df.index:
            df = df.drop("[CLS]")
        if "[SEP]" in df.index:
            df = df.drop("[SEP]")

        # 过滤掉不需要的聚合行和列
        df = df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        df = df.drop(columns=['support'], errors='ignore')

        df = df.rename(columns={
            'precision': 'P',
            'recall': 'R',
            'f1-score': 'F1'
        })
        df = df.round(4)

        # 将DataFrame转换为字符串
        report_str = df.to_string()
        lines.append(report_str)

    return "\n".join(lines)


def save_metrics_to_file(metrics: Dict[str, float], filepath: str) -> None:
    """
    将评估指标保存到文件

    Args:
        metrics: 评估指标字典
        filepath: 保存文件路径
    """
    formatted_metrics = format_metrics(metrics)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(formatted_metrics)

    logger.info(f"Evaluation metrics saved to {filepath}")
