import os
from pathlib import Path

def extract_filenames():
    """提取CMIN-CN/raw中所有CSV文件的名称"""
    # 设置目录路径
    data_root = Path("Data")
    cn_news_path = data_root / "CMIN-CN" / "news" / "raw"
    
    # 存储文件名的列表
    filenames = []
    
    try:
        # 遍历目录获取所有csv文件
        for file in cn_news_path.glob("*.csv"):
            # 添加文件名（不含后缀）到列表
            filenames.append(file.stem)
        
        # 排序文件名
        filenames.sort()
        
        # 打印文件名
        print("文件名列表：")
        for name in filenames:
            print(name)
        
        # 打印统计信息
        print(f"\n共找到 {len(filenames)} 个文件")
        
        # 可选：保存到文本文件
        output_file = "stock_names.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for name in filenames:
                f.write(name + "\n")
        print(f"\n文件名已保存到：{output_file}")
        
    except Exception as e:
        print(f"发生错误：{str(e)}")

if __name__ == "__main__":
    extract_filenames()