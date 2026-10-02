import spacy
import os
import re

# 初始化 spaCy 英文分词器
nlp = spacy.load("en_core_web_sm")


def read_text_file(filename):
    """ 读取原始文本文件并返回文本内容 """
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().strip()


def read_a2_file(filename):
    """ 读取A2标注文件并返回所有的触发词信息 """
    triggers = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("T"):  # 仅处理触发词标注（以"T"开头）
                    parts = line.split("\t")
                    trigger_id = parts[0]  # TID
                    trigger_info = parts[1].split()  # 以空格拆分

                    trigger_type = trigger_info[0]  # 触发词类型
                    start = int(trigger_info[1].strip())  # 起始字符位置
                    end = int(trigger_info[2].strip())  # 结束字符位置

                    triggers.append((start, end, trigger_type, trigger_id))
    except Exception as e:
        print(f"Error reading A2 file {filename}: {e}")
    return triggers


def tokenize_with_offsets(text):
    """ 使用 spaCy 分词，并返回 token 及其在文本中的起始和结束位置 """
    doc = nlp(text)
    token_offsets = []
    for token in doc:
        token_offsets.append((token.text, token.idx, token.idx + len(token.text)))
    return token_offsets


def create_bio_labels(text, triggers):
    """ 创建 BIO 标签，将文本中的触发词转换为BIO标签 """
    tokens_with_offsets = tokenize_with_offsets(text)
    tokens = [token for token, _, _ in tokens_with_offsets]
    labels = ['O'] * len(tokens)

    # 根据触发词的起始和结束位置，将相应的 token 标记为 B-<trigger> 或 I-<trigger>
    for start, end, trigger_type, _ in triggers:
        is_first_token = True
        for idx, (token, token_start, token_end) in enumerate(tokens_with_offsets):
            if token_end > start and token_start < end:  # token 在触发词范围内
                if is_first_token:
                    labels[idx] = f'B-{trigger_type}'
                    is_first_token = False
                else:
                    labels[idx] = f'I-{trigger_type}'

    return tokens, labels


def save_bio_format(tokens, labels, output_filename):
    """ 将 BIO 格式保存为文件 """
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            for token, label in zip(tokens, labels):
                # 仅保存非空白的 'O' 标注行
                if label == 'O' and token.strip() == '':  # 空行或空字符
                    continue
                f.write(f"{token}\t{label}\n")
            f.write("\n")
    except Exception as e:
        print(f"Error saving BIO file {output_filename}: {e}")


def merge_bio_files(output_dir, merged_filename):
    """ 将所有生成的 BIO 文件合并为一个文件，每个文件之间添加空行 """
    try:
        with open(merged_filename, 'w', encoding='utf-8') as merged_file:
            for filename in os.listdir(output_dir):
                if filename.endswith('.bio'):
                    bio_filename = os.path.join(output_dir, filename)
                    with open(bio_filename, 'r', encoding='utf-8') as f:
                        # 将当前文件内容写入合并文件，并在文件之间添加空行
                        merged_file.write(f.read())
                        # merged_file.write("\n")  # 文件之间添加换行
    except Exception as e:
        print(f"Error merging BIO files: {e}")


def process_files(input_dir, output_dir):
    """ 处理所有文本文件并生成对应的 BIO 格式文件 """
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            text_filename = os.path.join(input_dir, filename)
            a2_filename = os.path.join(input_dir, filename.replace('.txt', '.a2'))

            if os.path.exists(a2_filename):
                print(f"Processing {filename}...")
                text = read_text_file(text_filename)
                triggers = read_a2_file(a2_filename)

                if not triggers:
                    print(f"No triggers found in {a2_filename}, skipping file.")
                    continue

                # 创建 BIO 标签
                tokens, bio_labels = create_bio_labels(text, triggers)

                # 保存 BIO 格式
                output_filename = os.path.join(output_dir, filename.replace('.txt', '.bio'))
                save_bio_format(tokens, bio_labels, output_filename)
                print(f"BIO format file saved as {output_filename}")
            else:
                print(f"No corresponding A2 file for {filename}, skipping.")


if __name__ == '__main__':
    input_directory = './kb/datasets/bionlp13/devel'  # 输入文件夹路径
    output_directory = './kb/datasets/bionlp13/devel_toBIO'  # 输出文件夹路径
    merged_file = './kb/datasets/bionlp13/devel_toBIO/merged.bio'  # 合并后的文件路径

    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 处理所有文件
    process_files(input_directory, output_directory)

    # 合并所有的 BIO 文件
    merge_bio_files(output_directory, merged_file)
    print(f"All BIO files have been merged into {merged_file}")
