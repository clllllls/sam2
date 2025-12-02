import os
import sys


def generate_directory_tree(start_path, output_file):
    """
    生成目录树并写入到输出文件中
    :param start_path: 要遍历的根目录
    :param output_file: 输出文件对象
    """
    # 写入标题和根目录信息
    abs_start_path = os.path.abspath(start_path)
    output_file.write(f"目录结构: {abs_start_path}\n")
    output_file.write(f"生成时间: {get_current_time()}\n\n")

    # 递归遍历目录
    for root, dirs, files in os.walk(start_path):
        # 计算当前层级
        level = root.replace(start_path, '').count(os.sep)
        indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''

        # 写入当前目录
        dir_name = os.path.basename(root)
        if level == 0:  # 根目录特殊处理
            output_file.write(f'{dir_name}/\n')
        else:
            output_file.write(f'{indent}{dir_name}/\n')

        # 写入文件
        sub_indent = '│   ' * level + '├── '
        for i, f in enumerate(sorted(files)):
            # 判断是否是最后一个条目
            is_last = (i == len(files) - 1) and (not dirs)
            indent_char = '└── ' if is_last else '├── '
            file_indent = '│   ' * level + indent_char

            output_file.write(f'{file_indent}{f}\n')


def get_current_time():
    """获取当前时间字符串"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    # 参数处理
    target_dir = os.getcwd()  # 默认使用当前目录
    output_filename = "directory_structure.txt"

    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_filename = sys.argv[2]

    # 验证目录是否存在
    if not os.path.isdir(target_dir):
        print(f"错误: 目录不存在 - {target_dir}")
        sys.exit(1)

    # 生成目录树
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            generate_directory_tree(target_dir, f)
        print(f"成功生成目录结构到: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"生成失败: {str(e)}")