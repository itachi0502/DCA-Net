def _hybrid_sampling(self, low_freq_pairs, num_samples):
    """混合采样方法，结合随机、频率和上下文信息"""
    pairs = list(low_freq_pairs.keys())
    if len(pairs) <= num_samples:
        return [(pair, self._find_sentence_context(pair)) for pair in pairs]

    # 计算采样权重：结合频率的倒数（更稀有的更可能被选中）和一定的随机性
    weights = []
    contexts = []
    valid_pairs = []

    for pair in pairs:
        # 获取上下文信息
        context = self._find_sentence_context(pair)
        if context:  # 只考虑有上下文的对
            count, prob = low_freq_pairs[pair]
            # 权重 = 1/(频率+1) + 小随机因子
            weight = 1.0 / (count + 1) + random.random() * 0.1
            weights.append(weight)
            contexts.append(context)
            valid_pairs.append(pair)

    if not valid_pairs:
        logger.warning("没有找到有上下文的触发词对，回退到随机采样")
        return self._random_sampling(low_freq_pairs, num_samples)

    # 归一化权重
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    else:
        weights = [1.0 / len(weights)] * len(weights)

    # 根据权重采样，但不放回
    sample_indices = []
    for _ in range(min(num_samples, len(valid_pairs))):
        # 如果已经采样了所有可能的索引，就退出
        if len(sample_indices) == len(valid_pairs):
            break

        # 计算剩余项的权重
        remaining_indices = [i for i in range(len(valid_pairs)) if i not in sample_indices]
        remaining_weights = [weights[i] for i in remaining_indices]

        # 归一化剩余权重
        total_remaining = sum(remaining_weights)
        if total_remaining > 0:
            remaining_weights = [w / total_remaining for w in remaining_weights]
        else:
            remaining_weights = [1.0 / len(remaining_indices)] * len(remaining_indices)

        # 采样一个索引
        chosen_idx = np.random.choice(remaining_indices, p=remaining_weights)
        sample_indices.append(chosen_idx)

    # 构建结果
    sampled_pairs = [(valid_pairs[i], contexts[i]) for i in sample_indices]
    logger.info(f"通过混合采样选择了 {len(sampled_pairs)} 个触发词对")

    return sampled_pairs
import os
import re
import random
import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI  # 使用新的OpenAI客户端方式
import anthropic
from pathlib import Path
import yaml
import logging
from typing import List, Dict, Tuple, Set, Optional
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("biomedical_augmentation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class to manage settings"""

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
    def generation_model(self):
        return self.config.get('generation_model', 'chatgpt')

    @property
    def sampling_method(self):
        return self.config.get('sampling_method', 'markov')

    @property
    def api_keys(self):
        return self.config.get('api_keys', {})

    @property
    def mirror_services(self):
        return self.config.get('mirror_services', {})

    @property
    def output_path(self):
        return self.config.get('output_path', './augmented_data')

    @property
    def biobert_model(self):
        return self.config.get('biobert_model', './biobert')

    @property
    def augmentation_ratio(self):
        return self.config.get('augmentation_ratio', 0.3)

    @property
    def low_frequency_threshold(self):
        return self.config.get('low_frequency_threshold', 5)


class CONLLDataset:
    """Class to handle CONLL formatted BIO-tagged datasets"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.sentences = []
        self.labels = []
        self.trigger_words = set()
        self.trigger_pairs = defaultdict(int)
        self.load_data()

    def load_data(self):
        """Load data from CONLL format file"""
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

        # Add the last sentence if the file doesn't end with an empty line
        if current_sentence:
            self.sentences.append(current_sentence)
            self.labels.append(current_labels)

        logger.info(f"Loaded {len(self.sentences)} sentences from {self.file_path}")
        self._extract_triggers()

    def _extract_triggers(self):
        """Extract trigger words and their relationships from the dataset"""
        for sentence, labels in zip(self.sentences, self.labels):
            # Extract trigger words
            triggers = []
            current_trigger = []
            current_type = ""

            for token, label in zip(sentence, labels):
                if label.startswith("B-"):
                    if current_trigger:
                        triggers.append((" ".join(current_trigger), current_type))
                        current_trigger = []
                    current_trigger.append(token)
                    current_type = label[2:]  # Remove "B-" prefix
                elif label.startswith("I-") and current_trigger:
                    current_trigger.append(token)
                elif current_trigger:
                    triggers.append((" ".join(current_trigger), current_type))
                    current_trigger = []
                    current_type = ""

            # Add the last trigger if exists
            if current_trigger:
                triggers.append((" ".join(current_trigger), current_type))

            # Add to set of trigger words
            for trigger, trigger_type in triggers:
                self.trigger_words.add((trigger, trigger_type))

            # Count co-occurrences of trigger pairs
            for i in range(len(triggers)):
                for j in range(i + 1, len(triggers)):
                    trigger_pair = (triggers[i], triggers[j])
                    self.trigger_pairs[trigger_pair] += 1

        logger.info(f"Extracted {len(self.trigger_words)} unique trigger words")
        logger.info(f"Found {len(self.trigger_pairs)} trigger word pairs")

    def get_sentence_with_triggers(self):
        """Get sentences with their trigger words highlighted"""
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
        """Save dataset to CONLL format file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for sentence, labels in zip(self.sentences, self.labels):
                for token, label in zip(sentence, labels):
                    f.write(f"{token} {label}\n")
                f.write("\n")
        logger.info(f"Saved {len(self.sentences)} sentences to {file_path}")


class JointProbabilityCalculator:
    """Class to calculate joint probabilities of trigger pairs"""

    def __init__(self, dataset: CONLLDataset):
        self.dataset = dataset
        self.trigger_counts = Counter()
        self.pair_counts = Counter()
        self.total_triggers = 0
        self.joint_probabilities = {}
        self.calculate_probabilities()

    def calculate_probabilities(self):
        """Calculate marginal and joint probabilities of triggers"""
        # Count individual triggers
        for trigger, _ in self.dataset.trigger_words:
            self.trigger_counts[trigger] += 1
            self.total_triggers += 1

        # Count trigger pairs
        for (trigger1, type1), (trigger2, type2) in self.dataset.trigger_pairs:
            pair = (trigger1, trigger2)
            self.pair_counts[pair] += 1

        # Calculate joint probabilities
        for pair, count in self.pair_counts.items():
            trigger1, trigger2 = pair
            prob = count / self.total_triggers
            self.joint_probabilities[pair] = prob

        logger.info(f"Calculated joint probabilities for {len(self.joint_probabilities)} trigger pairs")

    def get_low_frequency_pairs(self, threshold):
        """Get trigger pairs with frequency below threshold"""
        low_freq_pairs = {}
        for pair, count in self.pair_counts.items():
            if count < threshold:
                low_freq_pairs[pair] = (count, self.joint_probabilities[pair])

        return low_freq_pairs


class BioBERTContextExtractor:
    """Class to extract contextual embeddings using BioBERT"""

    def __init__(self, model_name="dmis-lab/biobert-v1.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

        logger.info(f"Initialized BioBERT context extractor with device: {self.device}")

    def get_context_embedding(self, sentence, trigger_indices):
        """Extract contextual embedding for a trigger in a sentence"""
        tokens = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state[0]

        # Get the embedding for the trigger tokens
        trigger_embeddings = []
        for idx in trigger_indices:
            if idx < len(embeddings):
                trigger_embeddings.append(embeddings[idx])

        if trigger_embeddings:
            # Average the embeddings for multi-token triggers
            trigger_embedding = torch.mean(torch.stack(trigger_embeddings), dim=0)
            return trigger_embedding.cpu().numpy()
        else:
            # Return zeros if no valid indices found
            return np.zeros(embeddings.shape[1])

    def get_context_similarity(self, sentence1, trigger_indices1, sentence2, trigger_indices2):
        """Calculate cosine similarity between two trigger contexts"""
        emb1 = self.get_context_embedding(sentence1, trigger_indices1)
        emb2 = self.get_context_embedding(sentence2, trigger_indices2)

        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity


class TriggerSampler:
    """Class to sample trigger pairs using different methods"""

    def __init__(self, dataset: CONLLDataset, calculator: JointProbabilityCalculator,
                 extractor: BioBERTContextExtractor, method="markov"):
        self.dataset = dataset
        self.calculator = calculator
        self.extractor = extractor
        self.method = method
        self.sentences_with_triggers = dataset.get_sentence_with_triggers()

        logger.info(f"Initialized trigger sampler with method: {method}")

    def sample_trigger_pairs(self, num_samples, threshold):
        """Sample trigger pairs based on configured method"""
        low_freq_pairs = self.calculator.get_low_frequency_pairs(threshold)

        if not low_freq_pairs:
            logger.warning("No low frequency pairs found to sample from")
            return []

        logger.info(f"找到 {len(low_freq_pairs)} 个低频触发词对，使用 {self.method} 采样方法")

        try:
            if self.method == "random":
                return self._random_sampling(low_freq_pairs, num_samples)
            elif self.method == "frequency":
                return self._frequency_sampling(low_freq_pairs, num_samples)
            elif self.method == "markov":
                return self._markov_sampling(low_freq_pairs, num_samples)
            elif self.method == "cluster":
                return self._cluster_sampling(low_freq_pairs, num_samples)
            elif self.method == "hybrid":
                # 新增混合采样方法，结合随机和频率采样
                return self._hybrid_sampling(low_freq_pairs, num_samples)
            else:
                logger.warning(f"未知的采样方法: {self.method}，使用随机采样代替")
                return self._random_sampling(low_freq_pairs, num_samples)
        except Exception as e:
            logger.error(f"采样过程中出错: {str(e)}")
            logger.warning(f"回退到随机采样方法")
            return self._random_sampling(low_freq_pairs, num_samples)

    def _random_sampling(self, low_freq_pairs, num_samples):
        """Simple random sampling from low frequency pairs"""
        pairs = list(low_freq_pairs.keys())
        if len(pairs) <= num_samples:
            return [(pair, self._find_sentence_context(pair)) for pair in pairs]

        sampled_pairs = random.sample(pairs, num_samples)
        return [(pair, self._find_sentence_context(pair)) for pair in sampled_pairs]

    def _frequency_sampling(self, low_freq_pairs, num_samples):
        """Sample based on inverse frequency (rarer pairs more likely)"""
        pairs = list(low_freq_pairs.keys())
        weights = [1.0 / (count + 1) for _, (count, _) in low_freq_pairs.items()]

        # Normalize weights
        total = sum(weights)
        if total == 0:
            return self._random_sampling(low_freq_pairs, num_samples)

        weights = [w / total for w in weights]

        if len(pairs) <= num_samples:
            return [(pair, self._find_sentence_context(pair)) for pair in pairs]

        sampled_indices = np.random.choice(len(pairs), size=num_samples, replace=False, p=weights)
        sampled_pairs = [pairs[i] for i in sampled_indices]

        return [(pair, self._find_sentence_context(pair)) for pair in sampled_pairs]

    def _markov_sampling(self, low_freq_pairs, num_samples):
        """Markov chain Monte Carlo sampling considering context"""
        pairs = list(low_freq_pairs.keys())
        if not pairs:
            return []

        # Start with a random pair
        current_pair = random.choice(pairs)
        sampled_pairs = []

        # Use MCMC to sample pairs
        for _ in range(num_samples):
            # Propose a new pair
            proposed_pair = random.choice(pairs)

            # Find sentence contexts
            current_context = self._find_sentence_context(current_pair)
            proposed_context = self._find_sentence_context(proposed_pair)

            if not current_context or not proposed_context:
                current_pair = proposed_pair
                continue

            # Calculate acceptance probability based on joint probability and context similarity
            current_prob = low_freq_pairs[current_pair][1]  # Joint probability
            proposed_prob = low_freq_pairs[proposed_pair][1]

            # Extract context similarity if contexts are available
            context_weight = 0.5

            # Get indices for triggers in sentences
            current_indices1 = self._get_indices(current_context["sentence"], current_pair[0])
            current_indices2 = self._get_indices(current_context["sentence"], current_pair[1])
            proposed_indices1 = self._get_indices(proposed_context["sentence"], proposed_pair[0])
            proposed_indices2 = self._get_indices(proposed_context["sentence"], proposed_pair[1])

            # Calculate context similarity
            current_context_score = self.extractor.get_context_similarity(
                " ".join(current_context["sentence"]), current_indices1,
                " ".join(current_context["sentence"]), current_indices2
            )

            proposed_context_score = self.extractor.get_context_similarity(
                " ".join(proposed_context["sentence"]), proposed_indices1,
                " ".join(proposed_context["sentence"]), proposed_indices2
            )

            # Combine probabilities and context scores
            current_score = (1 - context_weight) * current_prob + context_weight * current_context_score
            proposed_score = (1 - context_weight) * proposed_prob + context_weight * proposed_context_score

            # Calculate acceptance probability
            acceptance_prob = min(1.0, proposed_score / current_score if current_score > 0 else 1.0)

            # Accept or reject
            if random.random() < acceptance_prob:
                current_pair = proposed_pair

            sampled_pairs.append((current_pair, current_context))

        return sampled_pairs

    def _cluster_sampling(self, low_freq_pairs, num_samples):
        """基于上下文相似性进行聚类采样"""
        pairs = list(low_freq_pairs.keys())

        if len(pairs) <= num_samples:
            return [(pair, self._find_sentence_context(pair)) for pair in pairs]

        # 获取上下文
        pair_contexts = {}
        for pair in pairs:
            context = self._find_sentence_context(pair)
            if context:
                pair_contexts[pair] = context

        if not pair_contexts:
            logger.warning("未找到任何触发词对的上下文，回退到随机采样")
            return self._random_sampling(low_freq_pairs, num_samples)

        logger.info(f"获取到 {len(pair_contexts)} 个触发词对的上下文信息")

        # 使用一种更简单但健壮的聚类方法
        selected_pairs = []
        remaining_pairs = list(pair_contexts.keys())

        # 随机选择第一个元素
        if remaining_pairs:
            first_pair = random.choice(remaining_pairs)
            selected_pairs.append(first_pair)
            remaining_pairs.remove(first_pair)

        # 基于上下文重叠度选择其余的元素
        while len(selected_pairs) < num_samples and remaining_pairs:
            max_dissimilarity = -1
            most_dissimilar_pair = None

            for pair in remaining_pairs:
                # 计算与已选择对的上下文差异
                min_similarity = float('inf')

                for selected in selected_pairs:
                    # 计算上下文差异，使用词汇重叠作为简单度量
                    context1 = pair_contexts[pair]["sentence"]
                    context2 = pair_contexts[selected]["sentence"]

                    # 计算两个上下文的词汇重叠程度
                    words1 = set(context1)
                    words2 = set(context2)
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))

                    # Jaccard相似度
                    similarity = intersection / union if union > 0 else 0
                    min_similarity = min(min_similarity, similarity)

                # 选择与现有样本最不相似的
                if min_similarity < max_dissimilarity or max_dissimilarity == -1:
                    max_dissimilarity = min_similarity
                    most_dissimilar_pair = pair

            if most_dissimilar_pair:
                selected_pairs.append(most_dissimilar_pair)
                remaining_pairs.remove(most_dissimilar_pair)
            else:
                break

        logger.info(f"通过聚类采样选择了 {len(selected_pairs)} 个触发词对")
        return [(pair, pair_contexts[pair]) for pair in selected_pairs]

    def _find_sentence_context(self, trigger_pair):
        """Find a sentence containing both triggers in the pair"""
        trigger1, trigger2 = trigger_pair

        for entry in self.sentences_with_triggers:
            sentence = entry["sentence"]
            triggers = [t["text"] for t in entry["triggers"]]

            if trigger1 in triggers and trigger2 in triggers:
                return entry

        return None

    def _get_indices(self, sentence, trigger_text):
        """Get indices of a trigger in a sentence"""
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


class SentenceGenerator:
    """Class to generate new sentences containing sampled trigger pairs"""

    def __init__(self, model_name, api_keys, mirror_services=None):
        self.model_name = model_name
        self.api_keys = api_keys
        self.mirror_services = mirror_services or {}

        # 对所有模型都使用相同的镜像服务
        # 获取镜像服务配置
        openai_mirror = self.mirror_services.get("openai", {})
        base_url = os.environ.get("LLM_BASE_URL") or openai_mirror.get("base_url", "https://YOUR_LLM_ENDPOINT/v1")
        api_key = os.environ.get("LLM_API_KEY") or api_keys.get("openai", "")
        if not api_key or "YOUR_LLM_ENDPOINT" in base_url:
            raise ValueError("Set LLM_API_KEY and LLM_BASE_URL for your compatible provider.")

        # 使用镜像服务
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        # 根据不同模型设置相应的客户端类型
        if model_name.lower() in ["gpt-4", "gpt-3.5-turbo", "chatgpt", "gpt-4o", "gpt-4o-mini"]:
            self.client_type = "openai"
            self.actual_model_name = self.model_name  # 保持原样
        elif model_name.lower() in ["claude", "claude-2", "claude-3"]:
            self.client_type = "claude"
            self.actual_model_name = "claude"  # 镜像服务使用的模型名称
        elif model_name.lower() in ["grok", "grok-1", "grok-2", "grok-3"]:
            self.client_type = "grok"
            self.actual_model_name = "grok-3"  # 修正为镜像服务实际接受的名称
        elif model_name.lower() in ["deepseek-v3", "deepseek-chat"]:
            self.client_type = "deepseek-v3"
            self.actual_model_name = "deepseek-v3"  # 镜像服务使用的模型名称
        else:
            logger.warning(f"Unsupported model: {model_name}, falling back to GPT-4o-mini")
            self.client_type = "openai"
            self.model_name = "gpt-4o-mini"
            self.actual_model_name = "gpt-4o-mini"

        logger.info(f"Initialized sentence generator with model: {model_name} using mirror service")

        logger.info(f"Initialized sentence generator with model: {model_name}")

    def generate_sentences(self, trigger_pairs_with_context, num_sentences_per_pair=1):
        """Generate new sentences containing trigger pairs"""
        generated_sentences = []

        for (trigger_pair, context) in trigger_pairs_with_context:
            if not context:
                continue

            trigger1, trigger2 = trigger_pair
            original_sentence = " ".join(context["sentence"])

            # Create prompt for generation
            prompt = self._create_generation_prompt(
                trigger1, trigger2, original_sentence, num_sentences_per_pair
            )

            try:
                # 所有模型都使用相同的镜像服务接口
                response = self._generate_with_openai(prompt)

                # Parse and process generated sentences
                sentences = self._parse_generated_sentences(response, trigger1, trigger2)
                generated_sentences.extend(sentences)

            except Exception as e:
                logger.error(f"Error generating sentences: {str(e)}")

        return generated_sentences

    def _create_generation_prompt(self, trigger1, trigger2, original_sentence, num_sentences):
        """Create prompt for sentence generation"""
        prompt = f"""You are an expert in biomedical text generation. I need you to generate {num_sentences} new sentences that include the following two biomedical trigger words:
1. "{trigger1}"
2. "{trigger2}"

Here's an example of a sentence that uses these trigger words:
"{original_sentence}"

Requirements:
1. Each new sentence should use both trigger words exactly as provided.
2. The sentences should be different from the example.
3. Make sure the sentences are scientifically accurate and maintain the same biomedical context.
4. Keep the sentences to a similar length as the example.

Please format your response with one sentence per line, without numbering or additional information.
"""
        return prompt

    def _generate_with_openai(self, prompt):
        """使用OpenAI兼容的镜像服务生成句子"""
        # 只有在模型是grok且fallback为True时才使用备选模型
        use_fallback = self.client_type == "grok"

        # 定义一个模型备选列表，如果grok模型不可用，我们会尝试这些备选
        fallback_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"] if use_fallback else []
        current_model = self.actual_model_name

        # 尝试使用当前模型生成，如果失败且是grok模型，则尝试备选
        try:
            logger.info(f"尝试使用模型 {current_model} 生成内容")

            # 使用OpenAI客户端方式调用API
            response = self.client.chat.completions.create(
                model=current_model,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant specialized in biomedical text generation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            logger.info(f"成功使用模型 {current_model} 生成内容")
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"使用模型 {current_model} 生成内容失败: {error_msg}")

            # 如果不使用备选模型或错误与"无可用渠道"无关，直接返回错误
            if not use_fallback or "无可用渠道" not in error_msg:
                logger.error(f"生成失败，错误: {error_msg}")
                return f"Error generating content: {error_msg[:100]}..."

            # 尝试备选模型
            for i, fallback_model in enumerate(fallback_models):
                try:
                    logger.info(f"尝试使用备选模型 {fallback_model} (备选 {i + 1}/{len(fallback_models)})")

                    response = self.client.chat.completions.create(
                        model=fallback_model,
                        messages=[
                            {"role": "system",
                             "content": "You are a helpful assistant specialized in biomedical text generation."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1000
                    )

                    logger.info(f"成功使用备选模型 {fallback_model} 生成内容")
                    return response.choices[0].message.content

                except Exception as e2:
                    error_msg2 = str(e2)
                    logger.warning(f"使用备选模型 {fallback_model} 生成内容失败: {error_msg2}")

                    # 如果这是最后一个备选模型，记录错误并返回
                    if i == len(fallback_models) - 1:
                        logger.error(f"所有模型尝试失败，最后错误: {error_msg2}")
                        return f"Error generating content: {error_msg2[:100]}..."

                    # 否则继续尝试下一个备选模型
                    continue

            # 这里通常不会执行到
            return "Error: Failed to generate content with all models"

    def _generate_with_anthropic(self, prompt):
        """使用镜像服务生成句子 - Claude兼容接口"""
        try:
            # 使用OpenAI兼容的镜像服务调用Claude API
            response = self.client.chat.completions.create(
                model="claude",  # 固定使用"claude"作为模型名
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant specialized in biomedical text generation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Claude API through mirror: {str(e)}")
            return f"Error generating content: {str(e)[:100]}..."

    def _parse_generated_sentences(self, response, trigger1, trigger2):
        """Parse generated sentences and ensure they contain the trigger words"""
        sentences = []

        # Split response into individual sentences
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove numbering if present
            line = re.sub(r"^\d+\.?\s*", "", line)

            # Check if the sentence contains both triggers
            if trigger1.lower() in line.lower() and trigger2.lower() in line.lower():
                sentences.append(line)

        return sentences


class SentenceTagger:
    """Class to automatically tag generated sentences with BIO tags"""

    def __init__(self, dataset: CONLLDataset):
        self.dataset = dataset
        self.trigger_words = dataset.trigger_words
        self.trigger_types = self._extract_trigger_types()

        logger.info(f"Initialized sentence tagger with {len(self.trigger_types)} trigger types")

    def _extract_trigger_types(self):
        """Extract mapping of trigger words to their types"""
        trigger_types = {}
        for trigger, type_label in self.trigger_words:
            trigger_types[trigger.lower()] = type_label
        return trigger_types

    def tag_sentence(self, sentence):
        """Tag a generated sentence with BIO tags"""
        # Tokenize the sentence
        tokens = sentence.split()
        labels = ["O"] * len(tokens)

        # Find and tag all trigger words
        for i in range(len(tokens)):
            for j in range(len(tokens), i, -1):
                span = " ".join(tokens[i:j]).lower()

                # Check if the span is a known trigger word
                for trigger, trigger_type in self.trigger_types.items():
                    if span == trigger.lower():
                        # Tag as B for beginning
                        labels[i] = f"B-{trigger_type}"

                        # Tag as I for inside
                        for k in range(i + 1, j):
                            labels[k] = f"I-{trigger_type}"

                        break

        return tokens, labels

    def tag_sentences(self, sentences):
        """Tag multiple sentences"""
        tagged_sentences = []
        tagged_labels = []

        for sentence in sentences:
            tokens, labels = self.tag_sentence(sentence)
            tagged_sentences.append(tokens)
            tagged_labels.append(labels)

        return tagged_sentences, tagged_labels


class AugmentationPipeline:
    """Main pipeline to coordinate the augmentation process"""

    def __init__(self, config_path):
        self.config = Config(config_path)
        self.dataset = None
        self.calculator = None
        self.extractor = None
        self.sampler = None
        self.generator = None
        self.tagger = None

    def run(self):
        """Run the full augmentation pipeline"""
        logger.info("Starting data augmentation pipeline")

        # Step 1: Load dataset
        self.dataset = CONLLDataset(self.config.data_paths["train"])

        # Step 2: Calculate joint probabilities
        self.calculator = JointProbabilityCalculator(self.dataset)

        # Step 3: Initialize BioBERT extractor
        self.extractor = BioBERTContextExtractor(self.config.biobert_model)

        # Step 4: Sample low-frequency trigger pairs
        self.sampler = TriggerSampler(
            self.dataset,
            self.calculator,
            self.extractor,
            method=self.config.sampling_method
        )

        num_samples = int(len(self.dataset.sentences) * self.config.augmentation_ratio)
        sampled_pairs = self.sampler.sample_trigger_pairs(
            num_samples,
            self.config.low_frequency_threshold
        )

        logger.info(f"Sampled {len(sampled_pairs)} trigger pairs for augmentation")

        # Step 5: Generate new sentences
        self.generator = SentenceGenerator(
            self.config.generation_model,
            self.config.api_keys,
            self.config.mirror_services
        )

        generated_sentences = self.generator.generate_sentences(
            sampled_pairs,
            num_sentences_per_pair=1
        )

        logger.info(f"Generated {len(generated_sentences)} new sentences")

        # Step 6: Tag generated sentences
        self.tagger = SentenceTagger(self.dataset)
        tagged_sentences, tagged_labels = self.tagger.tag_sentences(generated_sentences)

        logger.info(f"Tagged {len(tagged_sentences)} sentences")

        # Step 7: Create augmented dataset
        augmented_dataset = CONLLDataset(self.config.data_paths["train"])
        augmented_dataset.sentences.extend(tagged_sentences)
        augmented_dataset.labels.extend(tagged_labels)

        # Save augmented dataset
        os.makedirs(self.config.output_path, exist_ok=True)
        augmented_train_path = os.path.join(self.config.output_path, "augmented_train.txt")
        augmented_dataset.save_to_file(augmented_train_path)

        # Copy validation and test files
        val_output_path = os.path.join(self.config.output_path, "val.txt")
        test_output_path = os.path.join(self.config.output_path, "test.txt")

        val_dataset = CONLLDataset(self.config.data_paths["val"])
        val_dataset.save_to_file(val_output_path)

        test_dataset = CONLLDataset(self.config.data_paths["test"])
        test_dataset.save_to_file(test_output_path)

        logger.info(f"Data augmentation completed. Files saved to {self.config.output_path}")
        logger.info(f"Original training set: {len(self.dataset.sentences)} sentences")
        logger.info(f"Augmented training set: {len(augmented_dataset.sentences)} sentences")

        return {
            "augmented_train_path": augmented_train_path,
            "val_path": val_output_path,
            "test_path": test_output_path,
            "original_count": len(self.dataset.sentences),
            "augmented_count": len(augmented_dataset.sentences)
        }


def create_default_config():
    """Create a default configuration file if none exists"""
    config = {
        "train_data": "./kb/datasets/bionlp11/train.txt",
        "val_data": "./kb/datasets/bionlp11/val.txt",
        "test_data": "./kb/datasets/bionlp11/test.txt",
        "output_path": "./augmented_data",
        "generation_model": "deepseek-v3",  # Options: gpt-4o-mini, gpt-4o, chatgpt, claude, grok, deepseek
        "sampling_method": "markov",  # Options: markov, random, frequency, cluster
        "biobert_model": "dmis-lab/biobert-v1.1",
        "augmentation_ratio": 0.3,
        "low_frequency_threshold": 5,
        "api_keys": {
            "openai": ""  # 统一的API密钥
        },
        "mirror_services": {
            "openai": {
                "base_url": "https://YOUR_LLM_ENDPOINT/v1",
                "default_model": "gpt-4o-mini"
            }
        }
    }

    with open("augmentation_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info("Created default configuration file: augmentation_config.yaml")
    return "augmentation_config.yaml"


def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description="Biomedical Trigger Detection Data Augmentation")
    parser.add_argument("--config", type=str, default="augmentation_config.yaml", help="Path to config file")
    parser.add_argument("--create-config", action="store_true", help="Create default config file")
    parser.add_argument("--generation-model", type=str, help="Override generation model in config")
    parser.add_argument("--sampling-method", type=str, help="Override sampling method in config")
    args = parser.parse_args()

    if args.create_config:
        config_path = create_default_config()
        logger.info(f"Created default config at {config_path}. Please update API keys before running.")
        return

    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        logger.info("Creating default config file...")
        args.config = create_default_config()

    # Load config
    config = Config(args.config)

    # Override config with command line arguments
    if args.generation_model:
        config.config["generation_model"] = args.generation_model

    if args.sampling_method:
        config.config["sampling_method"] = args.sampling_method

    # Save updated config
    config.save_config()

    # Run pipeline
    pipeline = AugmentationPipeline(args.config)
    results = pipeline.run()

    logger.info("Data augmentation completed successfully!")
    logger.info(f"Original dataset: {results['original_count']} sentences")
    logger.info(f"Augmented dataset: {results['augmented_count']} sentences")
    logger.info(f"Files saved to: {config.output_path}")


if __name__ == "__main__":
    main()
