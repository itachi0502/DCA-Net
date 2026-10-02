#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用MCMC采样和Llama生成模型进行生物医学触发词检测数据增强
"""

import os
import re
import random
import json
import numpy as np
import pandas as pd
import torch
import requests
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModel
import logging
from typing import List, Dict, Tuple, Set, Optional
import argparse
import yaml
import time
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mcmc_llama_augmentation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """配置类，管理实验设置"""

    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def save_config(self):
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

    @property
    def data_paths(self):
        return {
            'train': self.config.get('train_data'),
            'val': self.config.get('val_data'),
            'test': self.config.get('test_data')
        }

    @property
    def llama_api(self):
        return {
            'url': os.environ.get('LLAMA_API_URL') or self.config.get('llama_api', {}).get('url', "https://YOUR_LLM_ENDPOINT/v1/chat/completions"),
            'token': os.environ.get('LLAMA_API_KEY') or self.config.get('llama_api', {}).get('token', ''),
            'model': self.config.get('llama_api', {}).get('model', "llama-3.1-8B-instruct")
        }

    @property
    def output_path(self):
        return self.config.get('output_path', './augmented_data')

    @property
    def biobert_model(self):
        return self.config.get('biobert_model', 'dmis-lab/biobert-v1.1')

    @property
    def augmentation_ratio(self):
        return self.config.get('augmentation_ratio', 0.3)

    @property
    def low_frequency_threshold(self):
        return self.config.get('low_frequency_threshold', 5)

    @property
    def mcmc_iterations(self):
        return self.config.get('mcmc_iterations', 1000)

    @property
    def mcmc_burn_in(self):
        return self.config.get('mcmc_burn_in', 100)

    @property
    def mcmc_context_weight(self):
        return self.config.get('mcmc_context_weight', 0.6)


class CONLLDataset:
    """CONLL格式BIO标注数据集处理类"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.sentences = []
        self.labels = []
        self.trigger_words = set()
        self.trigger_pairs = defaultdict(int)
        self.load_data()

    def load_data(self):
        """加载CONLL格式数据"""
        current_sentence = []
        current_labels = []

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_sentence:
                        self.sentences.append(current_sentence)
                        self.labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        token, label = parts[0], parts[1]
                        current_sentence.append(token)
                        current_labels.append(label)

        # 添加最后一个句子（如果文件不以空行结束）
        if current_sentence:
            self.sentences.append(current_sentence)
            self.labels.append(current_labels)

        logger.info(f"从 {self.file_path} 加载了 {len(self.sentences)} 个句子")
        self._extract_triggers()

    def _extract_triggers(self):
        """从数据集中提取触发词及其关系"""
        for sentence, labels in zip(self.sentences, self.labels):
            # 提取触发词
            triggers = []
            current_trigger = []
            current_type = ""

            for token, label in zip(sentence, labels):
                if label.startswith("B-"):
                    if current_trigger:
                        triggers.append((" ".join(current_trigger), current_type))
                        current_trigger = []
                    current_trigger.append(token)
                    current_type = label[2:]  # 移除 "B-" 前缀
                elif label.startswith("I-") and current_trigger:
                    current_trigger.append(token)
                elif current_trigger:
                    triggers.append((" ".join(current_trigger), current_type))
                    current_trigger = []
                    current_type = ""

            # 添加最后一个触发词（如果存在）
            if current_trigger:
                triggers.append((" ".join(current_trigger), current_type))

            # 添加到触发词集合
            for trigger, trigger_type in triggers:
                self.trigger_words.add((trigger, trigger_type))

            # 计算触发词对的共现次数
            for i in range(len(triggers)):
                for j in range(i + 1, len(triggers)):
                    trigger_pair = (triggers[i], triggers[j])
                    self.trigger_pairs[trigger_pair] += 1

        logger.info(f"提取了 {len(self.trigger_words)} 个独特的触发词")
        logger.info(f"发现了 {len(self.trigger_pairs)} 个触发词对")

    def get_sentence_with_triggers(self):
        """获取带有触发词高亮的句子"""
        results = []

        for sentence, labels in zip(self.sentences, self.labels):
            triggers = []
            current_trigger = []
            current_type = ""
            current_indices = []

            for i, (token, label) in enumerate(zip(sentence, labels)):
                if label.startswith("B-"):
                    if current_trigger:
                        triggers.append({
                            "text": " ".join(current_trigger),
                            "type": current_type,
                            "indices": current_indices
                        })
                        current_trigger = []
                        current_indices = []
                    current_trigger.append(token)
                    current_indices.append(i)
                    current_type = label[2:]
                elif label.startswith("I-") and current_trigger:
                    current_trigger.append(token)
                    current_indices.append(i)
                elif current_trigger:
                    triggers.append({
                        "text": " ".join(current_trigger),
                        "type": current_type,
                        "indices": current_indices
                    })
                    current_trigger = []
                    current_indices = []
                    current_type = ""

            if current_trigger:
                triggers.append({
                    "text": " ".join(current_trigger),
                    "type": current_type,
                    "indices": current_indices
                })

            if triggers:
                results.append({
                    "sentence": sentence,
                    "triggers": triggers
                })

        return results

    def save_to_file(self, file_path):
        """保存数据集到CONLL格式文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for sentence, labels in zip(self.sentences, self.labels):
                for token, label in zip(sentence, labels):
                    f.write(f"{token} {label}\n")
                f.write("\n")
        logger.info(f"将 {len(self.sentences)} 个句子保存到 {file_path}")


class JointProbabilityCalculator:
    """计算触发词对联合概率的类"""

    def __init__(self, dataset: CONLLDataset):
        self.dataset = dataset
        self.trigger_counts = Counter()
        self.pair_counts = Counter()
        self.total_triggers = 0
        self.joint_probabilities = {}
        self.calculate_probabilities()

    def calculate_probabilities(self):
        """计算触发词的边缘和联合概率"""
        # 计算单个触发词的计数
        for trigger, _ in self.dataset.trigger_words:
            self.trigger_counts[trigger] += 1
            self.total_triggers += 1

        # 计算触发词对的计数
        for (trigger1, type1), (trigger2, type2) in self.dataset.trigger_pairs:
            pair = (trigger1, trigger2)
            self.pair_counts[pair] += 1

        # 计算联合概率
        for pair, count in self.pair_counts.items():
            trigger1, trigger2 = pair
            prob = count / self.total_triggers if self.total_triggers > 0 else 0
            self.joint_probabilities[pair] = prob

        logger.info(f"计算了 {len(self.joint_probabilities)} 个触发词对的联合概率")

    def get_low_frequency_pairs(self, threshold):
        """获取频率低于阈值的触发词对"""
        low_freq_pairs = {}
        for pair, count in self.pair_counts.items():
            if count < threshold:
                low_freq_pairs[pair] = (count, self.joint_probabilities[pair])

        return low_freq_pairs


class BioBERTContextExtractor:
    """使用BioBERT提取上下文嵌入的类"""

    def __init__(self, model_name="dmis-lab/biobert-v1.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

        logger.info(f"初始化BioBERT上下文提取器，使用设备: {self.device}")

    def get_context_embedding(self, sentence, trigger_indices):
        """提取句子中触发词的上下文嵌入"""
        tokens = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state[0]

        # 获取触发词token的嵌入
        trigger_embeddings = []
        for idx in trigger_indices:
            if idx < len(embeddings):
                trigger_embeddings.append(embeddings[idx])

        if trigger_embeddings:
            # 对多token触发词的嵌入取平均
            trigger_embedding = torch.mean(torch.stack(trigger_embeddings), dim=0)
            return trigger_embedding.cpu().numpy()
        else:
            # 如果没有有效的索引，返回零向量
            return np.zeros(embeddings.shape[1])

    def get_context_similarity(self, sentence1, trigger_indices1, sentence2, trigger_indices2):
        """计算两个触发词上下文之间的余弦相似度"""
        emb1 = self.get_context_embedding(sentence1, trigger_indices1)
        emb2 = self.get_context_embedding(sentence2, trigger_indices2)

        # 计算余弦相似度
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        return similarity


class MCMCSampler:
    """使用马尔可夫链蒙特卡洛方法采样触发词对的类"""

    def __init__(self, dataset: CONLLDataset, calculator: JointProbabilityCalculator,
                 extractor: BioBERTContextExtractor, iterations=1000, burn_in=100, context_weight=0.6):
        self.dataset = dataset
        self.calculator = calculator
        self.extractor = extractor
        self.iterations = iterations
        self.burn_in = burn_in
        self.context_weight = context_weight
        self.sentences_with_triggers = dataset.get_sentence_with_triggers()

        logger.info(f"初始化MCMC采样器，迭代次数={iterations}，预热期={burn_in}，上下文权重={context_weight}")

    def sample_trigger_pairs(self, num_samples, threshold):
        """使用MCMC采样低频触发词对"""
        low_freq_pairs = self.calculator.get_low_frequency_pairs(threshold)

        if not low_freq_pairs:
            logger.warning("没有找到可采样的低频触发词对")
            return []

        logger.info(f"找到 {len(low_freq_pairs)} 个低频触发词对")

        # 获取所有低频触发词对的上下文
        pair_contexts = {}
        for pair in low_freq_pairs.keys():
            context = self._find_sentence_context(pair)
            if context:
                pair_contexts[pair] = context

        if not pair_contexts:
            logger.warning("没有找到包含触发词对的上下文")
            return []

        logger.info(f"找到 {len(pair_contexts)} 个触发词对的上下文")

        # 开始MCMC采样
        pairs = list(pair_contexts.keys())

        # 如果可用对数量少于要求的样本数，直接返回所有
        if len(pairs) <= num_samples:
            return [(pair, pair_contexts[pair]) for pair in pairs]

        # 初始化随机选择一个触发词对
        current_pair = random.choice(pairs)
        current_context = pair_contexts[current_pair]

        # 存储采样结果
        samples = []
        accepted = 0

        # 运行MCMC
        for i in range(self.iterations):
            # 随机提出一个新的触发词对
            proposed_pair = random.choice(pairs)
            proposed_context = pair_contexts[proposed_pair]

            # 计算当前pair和proposed pair的概率
            current_score = self._calculate_score(current_pair, current_context, low_freq_pairs)
            proposed_score = self._calculate_score(proposed_pair, proposed_context, low_freq_pairs)

            # 计算接受概率
            acceptance_prob = min(1.0, proposed_score / (current_score + 1e-10))

            # 接受或拒绝
            if random.random() < acceptance_prob:
                current_pair = proposed_pair
                current_context = proposed_context
                accepted += 1

            # 预热期过后开始收集样本
            if i >= self.burn_in and i % max(1, (self.iterations - self.burn_in) // num_samples) == 0:
                if len(samples) < num_samples and current_pair not in [p for p, _ in samples]:
                    samples.append((current_pair, current_context))

        logger.info(f"MCMC采样完成，接受率: {accepted / self.iterations:.2f}，收集到 {len(samples)} 个样本")

        # 如果样本不足，随机补充
        if len(samples) < num_samples:
            remaining = [pair for pair in pairs if pair not in [p for p, _ in samples]]
            additional = min(num_samples - len(samples), len(remaining))

            if additional > 0:
                additional_pairs = random.sample(remaining, additional)
                for pair in additional_pairs:
                    samples.append((pair, pair_contexts[pair]))

                logger.info(f"额外随机补充了 {additional} 个样本")

        return samples[:num_samples]

    def _calculate_score(self, pair, context, low_freq_pairs):
        """计算触发词对的分数（结合联合概率和上下文信息）"""
        # 获取联合概率
        _, joint_prob = low_freq_pairs[pair]

        # 获取上下文相似性
        trigger1, trigger2 = pair

        # 获取触发词在句子中的位置
        indices1 = self._get_indices(context["sentence"], trigger1[0])
        indices2 = self._get_indices(context["sentence"], trigger2[0])

        # 计算上下文相似性
        if indices1 and indices2:
            sentence_text = " ".join(context["sentence"])
            context_similarity = self.extractor.get_context_similarity(
                sentence_text, indices1, sentence_text, indices2
            )
        else:
            context_similarity = 0.0

        # 计算加权分数：低频词对有更高的分数，相似度高的上下文有更高的分数
        score = (1 - self.context_weight) * (1.0 / (joint_prob + 1e-10)) + self.context_weight * context_similarity

        return max(score, 1e-10)  # 确保分数始终为正

    def _find_sentence_context(self, trigger_pair):
        """找到包含触发词对的句子，放宽匹配条件"""
        trigger1, trigger2 = trigger_pair

        # 首先尝试精确匹配
        for entry in self.sentences_with_triggers:
            sentence = entry["sentence"]
            trigger_texts = [t["text"] for t in entry["triggers"]]
            trigger_types = [t["type"] for t in entry["triggers"]]

            # 检查是否包含这两个特定类型的触发词
            if (trigger1[0] in trigger_texts and trigger2[0] in trigger_texts):
                idx1 = trigger_texts.index(trigger1[0])
                idx2 = trigger_texts.index(trigger2[0])

                # 确认类型也匹配
                if trigger_types[idx1] == trigger1[1] and trigger_types[idx2] == trigger2[1]:
                    return entry

        # 如果精确匹配没找到，尝试部分匹配（只匹配文本，忽略类型）
        for entry in self.sentences_with_triggers:
            sentence = entry["sentence"]
            trigger_texts = [t["text"] for t in entry["triggers"]]

            trigger1_text_lower = trigger1[0].lower()
            trigger2_text_lower = trigger2[0].lower()

            # 检查文本是否包含在任何触发词中
            has_trigger1 = any(trigger1_text_lower in t.lower() for t in trigger_texts)
            has_trigger2 = any(trigger2_text_lower in t.lower() for t in trigger_texts)

            if has_trigger1 and has_trigger2:
                return entry

        # 如果还是没找到，尝试在原始句子中查找关键词
        for entry in self.sentences_with_triggers:
            sentence_text = " ".join(entry["sentence"]).lower()
            if trigger1[0].lower() in sentence_text and trigger2[0].lower() in sentence_text:
                return entry

        return None

    def _get_indices(self, sentence, trigger_text):
        """获取触发词在句子中的索引"""
        if not trigger_text or not sentence:
            return []

        trigger_tokens = trigger_text.split()
        indices = []

        for i in range(len(sentence) - len(trigger_tokens) + 1):
            match = True
            for j in range(len(trigger_tokens)):
                if sentence[i + j] != trigger_tokens[j]:
                    match = False
                    break

            if match:
                indices.extend(range(i, i + len(trigger_tokens)))
                break

        return indices


class LlamaGenerator:
    """使用Llama生成新句子的类"""

    def __init__(self, api_config):
        self.api_url = api_config['url']
        self.api_token = api_config['token']
        if not self.api_token or 'YOUR_LLM_ENDPOINT' in self.api_url:
            raise ValueError('Set LLAMA_API_KEY and LLAMA_API_URL for your Llama-compatible provider.')
        self.model = api_config['model']
        self.retry_attempts = 3
        self.retry_delay = 2  # 秒

        logger.info(f"初始化Llama生成器，API URL: {self.api_url}, 模型: {self.model}")

    def generate_sentences(self, trigger_pairs_with_context, num_sentences_per_pair=1):
        """生成包含触发词对的新句子"""
        generated_sentences = []

        for i, (trigger_pair, context) in enumerate(trigger_pairs_with_context):
            if not context:
                continue

            trigger1, trigger2 = trigger_pair
            original_sentence = " ".join(context["sentence"])

            # 创建生成提示
            prompt = self._create_generation_prompt(
                trigger1[0], trigger2[0], trigger1[1], trigger2[1],
                original_sentence, num_sentences_per_pair
            )

            try:
                logger.info(f"为触发词对 {i + 1}/{len(trigger_pairs_with_context)} 生成新句子")

                # 生成句子
                response = self._generate_with_llama(prompt)

                # 解析生成的句子
                sentences = self._parse_generated_sentences(response, trigger1[0], trigger2[0])
                generated_sentences.extend(sentences)

                # 如果成功生成，记录信息
                if sentences:
                    logger.info(f"成功生成 {len(sentences)} 个新句子")
                else:
                    logger.warning(f"未能为触发词对 {trigger1[0]}-{trigger2[0]} 生成有效句子")

            except Exception as e:
                logger.error(f"生成句子时出错: {str(e)}")

        return generated_sentences

    def _create_generation_prompt(self, trigger1, trigger2, type1, type2, original_sentence, num_sentences):
        """创建生成提示"""
        prompt = f"""你是一位生物医学领域的专家，熟悉各种生物医学术语和触发词。

我需要你生成 {num_sentences} 个新的生物医学相关句子，每个句子都必须包含以下两个特定的触发词：
1. "{trigger1}" - 类型: {type1}
2. "{trigger2}" - 类型: {type2}

这是一个包含这两个触发词的示例句子：
"{original_sentence}"

要求：
1. 生成的句子必须包含两个触发词，准确使用它们的原始形式。
2. 句子应该具有科学准确性，并保持与示例句子类似的生物医学上下文。
3. 句子长度应与示例句子相近。
4. 句子应该与示例不同，但在同一生物医学领域。
5. 仅返回生成的句子，每行一个，不要添加编号或其他信息。

请生成具有科学价值的生物医学句子，确保两个触发词在句子中的使用方式符合它们的类型和生物医学含义。"""

        return prompt

    def _generate_with_llama(self, prompt):
        """使用Llama API生成内容"""
        messages = [
            {"role": "user", "content": prompt}
        ]

        logger.info(f"开始调用Llama API，提示长度: {len(prompt)}")

        for attempt in range(self.retry_attempts):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1000
                }

                logger.info(f"尝试 {attempt + 1}/{self.retry_attempts}: 发送请求到 {self.api_url}")

                # 发送请求
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=30  # 设置超时时间
                )

                logger.info(f"API响应状态码: {response.status_code}")

                if response.status_code != 200:
                    logger.error(f"API请求失败: {response.text}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        return f"Error: API returned status code {response.status_code}"

                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    generated_text = result["choices"][0]["message"]["content"]
                    logger.info(f"成功生成文本，长度: {len(generated_text)}")
                    return generated_text
                else:
                    logger.warning(f"API返回无效响应: {result}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                    else:
                        return "Error: Invalid response from API"

            except requests.exceptions.Timeout:
                logger.warning(f"API请求超时 (尝试 {attempt + 1}/{self.retry_attempts})")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * 2)  # 超时后等待更长时间
                else:
                    return "Error: API request timed out"

            except requests.exceptions.RequestException as e:
                logger.warning(f"API请求失败 (尝试 {attempt + 1}/{self.retry_attempts}): {str(e)}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    return f"Error: {str(e)}"

            except Exception as e:
                logger.error(f"调用API时发生未知错误: {str(e)}")
                return f"Error: Unknown error - {str(e)}"

        return "Error: Maximum retry attempts reached"

    def _parse_generated_sentences(self, response, trigger1, trigger2):
        """解析生成的句子并确保它们包含触发词"""
        if response.startswith("Error:"):
            logger.error(f"生成失败: {response}")
            return []

        sentences = []

        # 按行分割响应
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 移除编号（如果存在）
            line = re.sub(r"^\d+\.?\s*", "", line)

            # 检查句子是否包含两个触发词
            trigger1_lower = trigger1.lower()
            trigger2_lower = trigger2.lower()

            if trigger1_lower in line.lower() and trigger2_lower in line.lower():
                sentences.append(line)

        return sentences


class SentenceTagger:
    """自动为生成的句子添加BIO标签的类"""

    def __init__(self, dataset: CONLLDataset):
        self.dataset = dataset
        self.trigger_words = dataset.trigger_words
        self.trigger_types = self._extract_trigger_types()

        logger.info(f"初始化句子标注器，有 {len(self.trigger_types)} 种触发词类型")

    def _extract_trigger_types(self):
        """提取触发词到类型的映射"""
        trigger_types = {}
        for trigger, type_label in self.trigger_words:
            trigger_types[trigger.lower()] = type_label
        return trigger_types

    def tag_sentence(self, sentence):
        """为生成的句子添加BIO标签"""
        # 分词
        tokens = sentence.split()
        labels = ["O"] * len(tokens)

        # 查找并标注所有触发词
        for i in range(len(tokens)):
            for j in range(len(tokens), i, -1):
                span = " ".join(tokens[i:j]).lower()

                # 检查span是否是已知的触发词
                for trigger, trigger_type in self.trigger_types.items():
                    if span == trigger.lower():
                        # 开始标记为B
                        labels[i] = f"B-{trigger_type}"

                        # 内部标记为I
                        for k in range(i + 1, j):
                            labels[k] = f"I-{trigger_type}"

                        break

        return tokens, labels

    def tag_sentences(self, sentences):
        """标注多个句子"""
        tagged_sentences = []
        tagged_labels = []

        for sentence in sentences:
            tokens, labels = self.tag_sentence(sentence)
            tagged_sentences.append(tokens)
            tagged_labels.append(labels)

        return tagged_sentences, tagged_labels


class MCMCLlamaAugmentation:
    """使用MCMC采样和Llama生成进行数据增强的主流程"""

    def __init__(self, config_path):
        self.config = Config(config_path)
        self.dataset = None
        self.calculator = None
        self.extractor = None
        self.sampler = None
        self.generator = None
        self.tagger = None

    def run(self):
        """运行完整的数据增强流程"""
        logger.info("开始MCMC-Llama数据增强流程")

        # 步骤1: 加载数据集
        self.dataset = CONLLDataset(self.config.data_paths["train"])

        # 步骤2: 计算联合概率
        self.calculator = JointProbabilityCalculator(self.dataset)

        # 步骤3: 初始化BioBERT提取器
        self.extractor = BioBERTContextExtractor(self.config.biobert_model)

        # 步骤4: 使用MCMC采样低频触发词对
        self.sampler = MCMCSampler(
            self.dataset,
            self.calculator,
            self.extractor,
            iterations=self.config.mcmc_iterations,
            burn_in=self.config.mcmc_burn_in,
            context_weight=self.config.mcmc_context_weight
        )

        num_samples = int(len(self.dataset.sentences) * self.config.augmentation_ratio)
        sampled_pairs = self.sampler.sample_trigger_pairs(
            num_samples,
            self.config.low_frequency_threshold
        )

        logger.info(f"为增强采样了 {len(sampled_pairs)} 个触发词对")

        # 步骤5: 使用Llama生成新句子
        self.generator = LlamaGenerator(self.config.llama_api)

        generated_sentences = self.generator.generate_sentences(
            sampled_pairs,
            num_sentences_per_pair=1
        )

        logger.info(f"生成了 {len(generated_sentences)} 个新句子")

        # 步骤6: 标注生成的句子
        self.tagger = SentenceTagger(self.dataset)
        tagged_sentences, tagged_labels = self.tagger.tag_sentences(generated_sentences)

        logger.info(f"标注了 {len(tagged_sentences)} 个句子")

        # 步骤7: 创建增强数据集
        augmented_dataset = CONLLDataset(self.config.data_paths["train"])
        augmented_dataset.sentences.extend(tagged_sentences)
        augmented_dataset.labels.extend(tagged_labels)

        # 保存增强数据集
        os.makedirs(self.config.output_path, exist_ok=True)
        augmented_train_path = os.path.join(self.config.output_path, "augmented_train.txt")
        augmented_dataset.save_to_file(augmented_train_path)

        # 复制验证和测试文件
        val_output_path = os.path.join(self.config.output_path, "val.txt")
        test_output_path = os.path.join(self.config.output_path, "test.txt")

        val_dataset = CONLLDataset(self.config.data_paths["val"])
        val_dataset.save_to_file(val_output_path)

        test_dataset = CONLLDataset(self.config.data_paths["test"])
        test_dataset.save_to_file(test_output_path)

        logger.info(f"数据增强完成。文件保存到 {self.config.output_path}")
        logger.info(f"原始训练集: {len(self.dataset.sentences)} 个句子")
        logger.info(f"增强后训练集: {len(augmented_dataset.sentences)} 个句子")

        return {
            "augmented_train_path": augmented_train_path,
            "val_path": val_output_path,
            "test_path": test_output_path,
            "original_count": len(self.dataset.sentences),
            "augmented_count": len(augmented_dataset.sentences)
        }


def create_default_config():
    """创建默认配置文件（如果不存在）"""
    config = {
        "train_data": "./kb/datasets/bionlp13/train.txt",
        "val_data": "./kb/datasets/bionlp13/val.txt",
        "test_data": "./kb/datasets/bionlp13/test.txt",
        "output_path": "./bionlp13_llama_augmented_data",
        "biobert_model": "./biobert",
        "augmentation_ratio": 0.3,
        "low_frequency_threshold": 50,

        # MCMC相关配置
        "mcmc_iterations": 1000,
        "mcmc_burn_in": 100,
        "mcmc_context_weight": 0.6,

        # Llama API配置
        "llama_api": {
            "url": "https://YOUR_LLM_ENDPOINT/v1/chat/completions",
            "token": "",
            "model": "llama-3.1-8B-instruct"
        }
    }

    with open("mcmc_llama_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info("创建默认配置文件: mcmc_llama_config.yaml")
    return "mcmc_llama_config.yaml"


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用MCMC采样和Llama模型增强生物医学触发词检测数据")
    parser.add_argument("--config", type=str, default="mcmc_llama_config.yaml", help="配置文件路径")
    parser.add_argument("--create-config", action="store_true", help="创建默认配置文件")
    args = parser.parse_args()

    if args.create_config:
        config_path = create_default_config()
        logger.info(f"已创建默认配置文件 {config_path}。")
        return

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        logger.info("创建默认配置文件...")
        args.config = create_default_config()

    # 运行增强流程
    pipeline = MCMCLlamaAugmentation(args.config)
    results = pipeline.run()

    logger.info("数据增强成功完成!")
    logger.info(f"原始数据集: {results['original_count']} 个句子")
    logger.info(f"增强数据集: {results['augmented_count']} 个句子")
    logger.info(f"文件保存在: {results['augmented_train_path']}")


if __name__ == "__main__":
    main()
