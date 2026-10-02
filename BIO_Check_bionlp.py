import os

def check_bio_dataset(file_path):
    """检查BIO数据集格式"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    prev_tag = None
    errors = []

    for i, line in enumerate(lines):
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) != 2:
                errors.append(f"格式错误在第{i + 1}行：{line}")
                continue

            word, tag = parts
            if tag.startswith('I-') and (prev_tag is None or prev_tag[2:] != tag[2:] or prev_tag.startswith('O')):
                errors.append(f"I-标签不正确跟随在第{i + 1}行：{line}")

        prev_tag = tag if line else None

    if errors:
        print(f"文件 {file_path} 发现以下错误：")
        for error in errors:
            print(error)
    # else:
    #     print(f"文件 {file_path} BIO数据集格式检查通过，没有发现错误。")

def check_all_conll_files_in_directory(directory_path):
    """检查目录下所有的.conll文件"""
    for root, dirs, files in os.walk(directory_path):
        conll_files = [f for f in files if f.endswith('.bio')]  # 只选择 .conll 格式文件
        for conll_file in conll_files:
            file_path = os.path.join(root, conll_file)
            check_bio_dataset(file_path)

# 调用函数检查当前目录下所有.conll格式文件
directory_path = './kb/datasets/bionlp11/train_toBIO'
check_all_conll_files_in_directory(directory_path)
