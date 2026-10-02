import os
import re


def read_txt_file(file_path):
    """读取txt文件，返回文本内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_a2_file(file_path):
    """读取a2文件，返回触发词及其标注信息"""
    triggers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("T"):
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    trigger_id, event_info, trigger = parts
                    event_type, start, end = event_info.split()
                    triggers.append((trigger_id, event_type, int(start), int(end), trigger))
                elif len(parts) == 4:
                    trigger_id, event_info, trigger = parts
                    event_type, start, end = event_info.split()
                    triggers.append((trigger_id, event_type, int(start), int(end), trigger))
    return triggers


def convert_to_bio_format(text, triggers):
    """将文本和触发词转换为BIO标注格式"""
    words = re.findall(r'\S+', text)  # \S+ 匹配非空白字符序列
    bio_labels = ['O'] * len(words)  # 默认所有词标记为 'O'

    # 计算每个单词的字符位置（起始和结束字符位置）
    word_char_positions = []
    current_pos = 0
    for word in words:
        word_start = current_pos
        word_end = word_start + len(word)
        word_char_positions.append((word_start, word_end))
        current_pos = word_end + 1  # 加1表示空格或标点

    # 遍历触发词，标记BIO格式
    last_event_type = None  # 用于记录上一个触发词的事件类型，防止错误标记
    for trigger_id, event_type, start, end, trigger in triggers:
        trigger_start = None
        trigger_end = None

        # 寻找触发词在文本中的词级位置，并标记
        for i, (word_start, word_end) in enumerate(word_char_positions):
            if word_start <= start < word_end or word_start < end <= word_end:
                if trigger_start is None:
                    trigger_start = i  # 找到触发词开始的词位置
                trigger_end = i  # 更新触发词结束的位置

        # 如果触发词的索引无效，则跳过
        if trigger_start is None or trigger_end is None:
            print(f"Warning: Trigger {trigger} out of range. Skipping.")
            continue

        # 处理触发词的标记
        for i in range(trigger_start, trigger_end + 1):
            if bio_labels[i] == 'O':  # 只标记未标注的词
                if i == trigger_start:  # 第一个触发词的开始
                    # 如果是不同的事件类型，标记为B- {event_type}
                    if last_event_type != event_type:
                        bio_labels[i] = f'B-{event_type}'
                    else:
                        bio_labels[i] = f'B-{event_type}'
                else:  # 后续触发词
                    bio_labels[i] = f'I-{event_type}'

        # 更新 last_event_type，确保后续触发词属于当前事件类型
        last_event_type = event_type

    # 返回文本的单词和对应的BIO标注
    return words, bio_labels


def write_conll_format(output_path, words, bio_labels):
    """将转换后的BIO标注写入CoNLL格式的文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for word, label in zip(words, bio_labels):
            f.write(f"{word} {label}\n")
        f.write("\n")  # 每个文档后空一行，确保文档之间换行


def write_to_combined(output_path, words, bio_labels):
    """将每个文件的BIO标注结果追加到总汇文件"""
    with open(output_path, 'a', encoding='utf-8') as f:
        for word, label in zip(words, bio_labels):
            f.write(f"{word} {label}\n")
        f.write("\n")  # 每个文档后空一行，确保文档之间换行


def process_file(txt_file_path, a2_file_path, output_path, combined_file_path):
    """处理单个文档，读取txt和a2文件，转换为BIO格式"""
    # 读取txt文件内容
    text = read_txt_file(txt_file_path)

    # 读取a2文件中的触发词
    triggers = read_a2_file(a2_file_path)

    # 将文本和触发词转换为BIO格式
    words, bio_labels = convert_to_bio_format(text, triggers)

    # 写入CoNLL格式的输出文件
    write_conll_format(output_path, words, bio_labels)

    # 同时将每个文档的BIO标注结果追加到总汇文件
    write_to_combined(combined_file_path, words, bio_labels)


def process_dataset(dataset_dir, combined_file_path):
    """处理整个数据集，将每个文件转换为BIO格式，并生成总汇文件"""
    # 清空汇总文件，确保每次运行时不会重复添加内容
    with open(combined_file_path, 'w', encoding='utf-8') as f:
        pass

    # 获取文件夹下所有txt文件和对应的a2文件
    for root, dirs, files in os.walk(dataset_dir):
        txt_files = [f for f in files if f.endswith('.txt')]
        for txt_file in txt_files:
            # 构造对应的a2文件路径
            a2_file = txt_file.replace('.txt', '.a2')

            # 生成输出文件的路径（每个文档的单独文件）
            output_file = os.path.join(root, f"{txt_file.replace('.txt', '')}_bio.conll")

            # 处理每个txt文件与对应的a2文件
            txt_file_path = os.path.join(root, txt_file)
            a2_file_path = os.path.join(root, a2_file)
            process_file(txt_file_path, a2_file_path, output_file, combined_file_path)
            print(f"Processed {txt_file} -> {output_file}")


if __name__ == "__main__":
    dataset_dir = "./kb/datasets/bionlp11/train"
    combined_file_path = "./kb/datasets/bionlp11/train/combined_bio.conll"
    process_dataset(dataset_dir, combined_file_path)
