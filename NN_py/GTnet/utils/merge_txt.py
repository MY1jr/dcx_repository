from pathlib import Path

def merge_label_txt(txt1_path, txt2_path, output_path):
    # 确保输入路径是Path对象
    txt1_path = Path(txt1_path)
    txt2_path = Path(txt2_path)
    output_path = Path(output_path)

    # 检查输入文件是否存在
    if not txt1_path.exists():
        raise FileNotFoundError(f"找不到文件：{txt1_path}")
    if not txt2_path.exists():
        raise FileNotFoundError(f"找不到文件：{txt2_path}")

    # 读取两个文件
    lines1 = txt1_path.read_text().splitlines()
    lines2 = txt2_path.read_text().splitlines()

    # 确保两个文件行数相同
    if len(lines1) != len(lines2):
        raise ValueError("两个文件的行数不相同！")

    # 合并处理
    merged_lines = []
    for line1, line2 in zip(lines1, lines2):
        # 分割每行的内容
        parts1 = line1.split(',')
        parts2 = line2.split(',')
        
        # 合并：使用txt1的第一列，txt2的第一列作为第二列，保留原来的标签
        merged_line = f"{parts1[0]},{parts2[0]},{parts1[1]},{parts1[2]}\n"
        merged_lines.append(merged_line)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入新文件
    output_path.write_text(''.join(merged_lines))
    print(f"合并完成，结果已保存到：{output_path}")


def insert_dot_to_txt(txt_label_path, txt_dot_path, output_path):
    # 确保输入路径是Path对象
    txt_label_path = Path(txt_label_path)
    txt_dot_path = Path(txt_dot_path)
    output_path = Path(output_path)

    # 检查输入文件是否存在
    if not txt_label_path.exists():
        raise FileNotFoundError(f"找不到文件：{txt_label_path}")
    if not txt_dot_path.exists():
        raise FileNotFoundError(f"找不到文件：{txt_dot_path}")

    # 读取两个文件
    lines1 = txt_label_path.read_text().splitlines()
    lines2 = txt_dot_path.read_text().splitlines()

    # 确保两个文件行数相同
    if len(lines1) != len(lines2):
        raise ValueError("两个文件的行数不相同！")

    # 合并处理
    merged_lines = []
    for line1, line2 in zip(lines1, lines2):
        # 分割txt1的内容
        parts1 = line1.split(',')
        # txt2的每行就是一个数值
        value2 = line2.strip()
        
        # 合并：在原始行的倒数第三个位置插入txt2的值
        merged_parts = parts1[:-2] + [value2] + parts1[-2:]
        merged_line = ','.join(merged_parts) + '\n'
        merged_lines.append(merged_line)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入新文件
    output_path.write_text(''.join(merged_lines))
    print(f"合并完成，结果已保存到：{output_path}")


def merge_three_txt(txt1_path, txt2_path, txt3_path, output_path):
    """
    合并三个txt文件 ### 二合一
    
    参数:
        txt1_path: 第一个标签文件路径
        txt2_path: 第二个标签文件路径
        txt3_path: 点距离文件路径
        output_path: 最终输出文件路径
    """
    # 确保输入路径是Path对象
    txt1_path = Path(txt1_path)
    txt2_path = Path(txt2_path)
    txt3_path = Path(txt3_path)
    output_path = Path(output_path)

    # 检查输入文件是否存在
    if not txt1_path.exists():
        raise FileNotFoundError(f"找不到文件：{txt1_path}")
    if not txt2_path.exists():
        raise FileNotFoundError(f"找不到文件：{txt2_path}")
    if not txt3_path.exists():
        raise FileNotFoundError(f"找不到文件：{txt3_path}")

    # 读取所有文件
    lines1 = txt1_path.read_text().splitlines()
    lines2 = txt2_path.read_text().splitlines()
    lines3 = txt3_path.read_text().splitlines()

    # 确保文件行数相同
    if not (len(lines1) == len(lines2) == len(lines3)):
        raise ValueError("三个文件的行数不相同！")

    # 合并处理
    merged_lines = []
    for line1, line2, line3 in zip(lines1, lines2, lines3):
        # 第一步：合并txt1和txt2
        parts1 = line1.split(',')
        parts2 = line2.split(',')
        temp_merged = f"{parts1[0]},{parts2[0]},{parts1[1]},{parts1[2]}"
        
        # 第二步：将txt3的内容插入到临时合并结果中
        temp_parts = temp_merged.split(',')
        value3 = line3.strip()
        final_parts = temp_parts[:-2] + [value3] + temp_parts[-2:]
        merged_line = ','.join(final_parts) + '\n'
        
        merged_lines.append(merged_line)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入新文件
    output_path.write_text(''.join(merged_lines))
    print(f"三个文件合并完成，结果已保存到：{output_path}")

# # 测试
# if __name__ == "__main__":
#     # txt1_path = Path("G:/NN_py/ssstest_detected/ssstest_detected.txt")
#     # txt2_path = Path("G:/NN_py/ssstest_cluster/ssstest_cluster.txt")
#     # output_path = Path("G:/NN_py/ssstest_merge/ssstest_merge.txt")
#     # merge_label_txt(txt1_path, txt2_path, output_path)
    
#     txt_label_path = Path("G:/NN_py/ssstest_merge/ssstest_merge.txt")
#     txt_dot_path = Path("G:/NN_py/ssstest_distance/circle_distances.txt")
#     output_path = Path("G:/NN_py/ssstest_merge/ssstest_merge_dot.txt")
#     insert_dot_to_txt(txt_label_path, txt_dot_path, output_path)

# 二合一
if __name__ == "__main__":
    # 获取当前文件所在目录和项目根目录
    current_dir = Path(__file__).parent # utils
    project_root = current_dir.parent.parent # NN_py

    txt1_path = project_root / 'dcx' / 'dcx.txt' # label1
    txt2_path = project_root / 'dcx_cluster' / 'dcx_cluster.txt' # label2
    txt3_path = project_root / 'dcx_flow' / 'dcx_flow.txt' # distance
    output_path = project_root / 'dcx' / 'dcx_merge.txt'
    
    merge_three_txt(txt1_path, txt2_path, txt3_path, output_path)
