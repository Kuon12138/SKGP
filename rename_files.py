import os
from pathlib import Path

def rename_stock_files(directory: str = "Data/CMIN-CN/price/raw"):
    """重命名股票文件，将sz改为SZ，ss改为SH
    Args:
        directory: 包含股票文件的目录路径
    """
    try:
        # 转换为Path对象
        dir_path = Path(directory)
        
        # 检查目录是否存在
        if not dir_path.exists():
            raise ValueError(f"目录不存在: {directory}")
        
        # 获取所有csv文件
        files = list(dir_path.glob("*.csv"))
        
        if not files:
            print(f"警告: 在 {directory} 中没有找到CSV文件")
            return
        
        # 记录重命名操作
        renamed_count = 0
        
        # 遍历所有文件
        for file_path in files:
            old_name = file_path.name
            new_name = old_name
            
            # 替换文件扩展名前的后缀
            if '.sz.' in new_name.lower():
                new_name = new_name.lower().replace('.sz.', '.SZ.')
                renamed_count += 1
            elif '.ss.' in new_name.lower():
                new_name = new_name.lower().replace('.ss.', '.SH.')
                renamed_count += 1
            
            # 如果文件名需要更改
            if new_name != old_name:
                old_path = file_path
                new_path = file_path.parent / new_name
                
                # 检查新文件名是否已存在
                if new_path.exists():
                    print(f"警告: 文件已存在，跳过重命名: {new_name}")
                    continue
                
                try:
                    old_path.rename(new_path)
                    print(f"重命名: {old_name} -> {new_name}")
                except Exception as e:
                    print(f"重命名文件时出错 {old_name}: {str(e)}")
        
        print(f"\n完成重命名操作:")
        print(f"- 处理的文件总数: {len(files)}")
        print(f"- 重命名的文件数: {renamed_count}")
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    rename_stock_files()
