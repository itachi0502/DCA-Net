import logging
import os
import sys
from datetime import datetime


def setup_logger(config):
    """
    设置日志记录器

    Args:
        config: 配置信息

    Returns:
        logger: 配置好的日志记录器
    """
    # 获取配置
    log_level = config["logging"]["level"]
    log_format = config["logging"]["format"]
    to_file = config["logging"]["to_file"]
    to_console = config["logging"]["to_console"]

    # 创建日志目录
    log_dir = config["paths"]["log_dir"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建根日志记录器
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))

    # 清除现有处理器
    logger.handlers = []

    # 创建格式化器
    formatter = logging.Formatter(log_format)

    # 添加控制台处理器
    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 添加文件处理器
    if to_file:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(log_dir, f"ner_training_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 设置transformers库和flair库的日志级别
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("flair").setLevel(logging.WARNING)

    logger.info("Logger setup complete")
    logger.info(f"Log file: {log_file if to_file else 'N/A'}")

    return logger
