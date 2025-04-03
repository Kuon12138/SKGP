import pandas as pd
from pathlib import Path
import logging
from typing import List

class ExcelConverter:
    """Excel文件转换器"""
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def convert_xlsx_to_csv(self, input_file: str, output_file: str) -> None:
        """
        将xlsx文件转换为csv文件

        Args:
            input_file: 输入xlsx文件路径
            output_file: 输出csv文件路径
        """
        try:
            # 读取xlsx文件
            df = pd.read_excel(input_file)
            
            # 保存为csv
            df.to_csv(output_file, index=False, encoding='utf-8')
            self.logger.info(f"文件转换完成: {output_file}")
            
        except Exception as e:
            self.logger.error(f"文件转换失败 {input_file}: {str(e)}")
            raise

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        处理目录下的所有xlsx文件

        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有xlsx文件
        xlsx_files = list(input_path.glob("*.xlsx"))
        
        if not xlsx_files:
            self.logger.warning(f"在 {input_dir} 中没有找到xlsx文件")
            return
            
        self.logger.info(f"找到 {len(xlsx_files)} 个xlsx文件")
        
        # 转换每个文件
        for xlsx_file in xlsx_files:
            csv_file = output_path / f"{xlsx_file.stem}.csv"
            self.convert_xlsx_to_csv(str(xlsx_file), str(csv_file))

def main():
    """主函数"""
    # 设置数据根目录
    data_root = "Data"
    
    # 创建转换器
    converter = ExcelConverter(data_root)
    
    # 设置输入输出目录
    input_dir = Path(data_root) / "CMIN-CN" / "news" / "raw"
    output_dir = Path(data_root) / "CMIN-CN" / "news" / "raw"
    
    # 处理目录
    converter.process_directory(str(input_dir), str(output_dir))

if __name__ == "__main__":
    main() 