import os
import argparse
import yaml
import torch
import random
import numpy as np
from transformers import AutoTokenizer
import logging
from datetime import datetime

from src.simple.data_processor_s import get_data_loaders
from src.simple.train import NERTrainer
from src.simple.predictor import NERPredictor
from utils.logger import setup_logger
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed):
    """
    设置随机种子以确保可重复性

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train(config):
    """
    训练NER模型

    Args:
        config: 配置字典

    Returns:
        trainer: 训练好的模型训练器
        test_metrics: 测试集上的评估指标
    """
    # 设置随机种子
    set_seed(config["training"]["seed"])

    # 设置日志
    logger = setup_logger(config)

    # 记录训练开始
    logger.info("===== Starting NER Model Training =====")

    # 加载分词器
    logger.info(f"Loading tokenizer from {config['paths']['biobert_path']}")
    tokenizer = AutoTokenizer.from_pretrained(config["paths"]["biobert_path"])

    # 获取数据加载器和标签映射
    logger.info("Preparing data loaders...")
    train_loader, val_loader, test_loader, label_map, id_to_label = get_data_loaders(config, tokenizer)

    # 创建训练器
    logger.info("Creating trainer...")
    trainer = NERTrainer(config, train_loader, val_loader, test_loader, label_map, id_to_label)

    # 训练模型
    logger.info("Starting training...")
    test_metrics = trainer.train()

    # 记录训练结束
    logger.info("===== Training Complete =====")

    return trainer, test_metrics


def predict_test_set(trainer, config):
    """
    使用已训练的模型对测试集进行预测

    Args:
        trainer: 训练好的模型训练器
        config: 配置字典
    """
    # 获取最佳模型路径
    model_path = os.path.join(config["paths"]["model_save_dir"], "best_model")
    test_file = config["paths"]["test_data"]

    # 创建预测输出目录
    prediction_dir = config["paths"]["prediction_output"]
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(prediction_dir, f"test_predictions_{timestamp}.txt")

    # 创建预测器
    predictor = NERPredictor(model_path, config["training"]["device"])

    # 记录预测开始
    logging.info("===== Starting Test Set Prediction =====")
    logging.info(f"Model path: {model_path}")
    logging.info(f"Test file: {test_file}")
    logging.info(f"Output file: {output_file}")

    # 进行预测并评估
    metrics = predictor.evaluate_conll_file(test_file, output_file)

    # 记录预测结束
    logging.info("===== Test Set Prediction and Evaluation Complete =====")

    return metrics


def run_complete_pipeline(config_path="./config/config.yaml"):
    """
    运行完整的训练、验证、预测流程

    Args:
        config_path: 配置文件路径
    """
    # 加载配置
    config = load_config(config_path)

    # 训练模型
    trainer, train_metrics = train(config)

    # 预测测试集
    test_metrics = predict_test_set(trainer, config)

    print("\n===== Final Results =====")
    from utils.metrics_new import format_metrics
    print(format_metrics(test_metrics))

    print("\n训练、验证和预测已全部完成。")
    print(f"详细日志保存在: {config['paths']['log_dir']}")
    print(f"模型保存在: {os.path.join(config['paths']['model_save_dir'], 'best_model')}")
    print(f"预测结果保存在: {config['paths']['prediction_output']}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="NER Model Training and Prediction")
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["full", "train", "predict", "evaluate"], default="full",
                        help="Mode to run (full: train+validate+predict, train: only train, predict: only predict, evaluate: only evaluate)")
    parser.add_argument("--model_path", type=str,
                        help="Path to trained model directory (required for predict and evaluate modes)")
    parser.add_argument("--input_file", type=str, help="Path to input file (required for predict mode)")
    parser.add_argument("--output_file", type=str, help="Path to output file (required for predict mode)")
    parser.add_argument("--test_file", type=str, help="Path to test file (required for evaluate mode)")
    parser.add_argument("--device", type=str, help="Computation device (e.g., 'cuda:0', 'cpu')")

    args = parser.parse_args()

    if args.mode == "full" or args.mode is None:
        # 运行完整流程
        run_complete_pipeline(args.config)
    elif args.mode == "train":
        # 只进行训练
        config = load_config(args.config)
        train(config)
    elif args.mode == "predict":
        # 只进行预测
        if not args.model_path or not args.input_file or not args.output_file:
            print("Error: --model_path, --input_file, and --output_file are required for predict mode")
            return

        predictor = NERPredictor(args.model_path, args.device)
        predictor.predict_conll_file(args.input_file, args.output_file)
    elif args.mode == "evaluate":
        # 只进行评估
        if not args.model_path or not args.test_file:
            print("Error: --model_path and --test_file are required for evaluate mode")
            return

        predictor = NERPredictor(args.model_path, args.device)
        metrics = predictor.evaluate_conll_file(args.test_file, args.output_file)
        from utils.metrics_new import format_metrics
        print(format_metrics(metrics))


if __name__ == "__main__":
    main()
