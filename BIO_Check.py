# 检查BIO标签是否合规
def check_bio_dataset(file_path):
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
        print("发现以下错误：")
        for error in errors:
            print(error)
    else:
        print("BIO数据集格式检查通过，没有发现错误。")


# 调用函数检查BIO数据集
file_path = './kb/datasets/bionlp11/train.txt'
check_bio_dataset(file_path)
