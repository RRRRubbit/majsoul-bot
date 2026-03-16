import os
from pathlib import Path

def rename_images_in_folder(base_path):
    """
    批量重命名 tiles 文件夹下所有子文件夹中的图片
    格式：文件夹名_序号.扩展名
    """
    base_path = Path(base_path)

    if not base_path.exists():
        print(f"路径不存在: {base_path}")
        return

    # 遍历所有子文件夹
    subfolders = sorted([d for d in base_path.iterdir() if d.is_dir()])

    for subfolder in subfolders:
        folder_name = subfolder.name
        print(f"\n处理文件夹: {folder_name}")

        # 获取文件夹中所有图片文件（包括临时文件）
        all_files = []
        for file in subfolder.iterdir():
            if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                all_files.append(file.name)

        # 排序
        all_files = sorted(all_files)

        if not all_files:
            print(f"  没有找到图片文件")
            continue

        print(f"  找到 {len(all_files)} 个图片文件")

        # 第一阶段：将所有文件重命名为临时名称
        temp_names = []
        for idx, filename in enumerate(all_files, start=1):
            old_path = os.path.join(subfolder, filename)
            extension = Path(filename).suffix
            temp_name = f"TEMP_{idx:04d}{extension}"
            temp_path = os.path.join(subfolder, temp_name)

            try:
                os.rename(old_path, temp_path)
                temp_names.append((temp_name, idx, extension))
                print(f"  [阶段1] {filename} -> {temp_name}")
            except Exception as e:
                print(f"  [错误] 无法重命名 {filename}: {e}")

        # 第二阶段：将临时文件重命名为最终名称（统一为 .png 格式）
        for temp_name, idx, extension in temp_names:
            temp_path = os.path.join(subfolder, temp_name)
            new_name = f"{folder_name}_{idx:04d}.png"  # 统一使用 .png 扩展名
            new_path = os.path.join(subfolder, new_name)

            try:
                os.rename(temp_path, new_path)
                print(f"  [阶段2] {temp_name} -> {new_name}")
            except Exception as e:
                print(f"  [错误] 无法重命名 {temp_name}: {e}")

if __name__ == "__main__":
    tiles_path = r"D:\majsoul-bot\templates\tiles"
    print(f"开始重命名 {tiles_path} 中的图片文件...\n")
    rename_images_in_folder(tiles_path)
    print("\n重命名完成!")
