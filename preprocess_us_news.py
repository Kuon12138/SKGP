import logging
import os
import pandas as pd
from pathlib import Path
from typing import Optional

class USNewsPreprocessor:
    def __init__(self, data_root: str):
        """初始化US新闻数据预处理器
        Args:
            data_root: 数据根目录路径
        """
        self.data_root = Path(data_root)
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def preprocess_news_data(self, input_file: str, output_file: str) -> None:
        """处理US新闻数据
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        try:
            # 读取制表符分隔的CSV文件
            try:
                df = pd.read_csv(input_file, sep='\t')
            except pd.errors.ParserError:
                df = pd.read_csv(input_file, engine='python', on_bad_lines='skip', sep='\t')
            
            if df.empty:
                raise ValueError(f"空数据文件: {input_file}")
            
            # 删除不需要的列
            columns_to_drop = ['time', 'ticker', 'name', 'link']
            for col in columns_to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])
            
            # 创建新的数据框，指定列的顺序
            new_df = pd.DataFrame(columns=['publishedAt', 'title', 'description'])
            
            # 处理日期列
            new_df['publishedAt'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # 只保留2018-2021年的数据
            new_df = new_df[new_df['publishedAt'].str.startswith(('2018', '2019', '2020', '2021'))]
            
            # 添加标题和描述（交换内容）
            new_df['description'] = df['title'].astype(str).str.strip()
            new_df['title'] = df['summary'].astype(str).str.strip()
            
            # 按日期排序并删除重复行
            new_df = new_df.sort_values('publishedAt').reset_index(drop=True)
            new_df = new_df.drop_duplicates().reset_index(drop=True)
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存处理后的数据
            new_df.to_csv(output_file, index=False)
            self.logger.info(f"US新闻数据处理完成: {output_file}")
            
        except Exception as e:
            self.logger.error(f"US新闻数据处理失败: {str(e)}")
            self.logger.error(f"文件路径: {input_file}")
            raise

    def process_directory(self) -> None:
        """处理US新闻数据目录"""
        try:
            # 构建目录路径
            us_dir = self.data_root / "CMIN-US"
            news_raw_dir = us_dir / "news" / "raw"
            news_processed_dir = us_dir / "news" / "processed"
            
            # 创建处理后数据的目录
            os.makedirs(news_processed_dir, exist_ok=True)
            
            # 检查原始数据目录是否存在
            if not news_raw_dir.exists():
                raise ValueError(f"US新闻数据目录不存在: {news_raw_dir}")
            
            # 处理新闻数据
            news_files = list(news_raw_dir.glob("*.csv"))
            if not news_files:
                self.logger.warning(f"没有找到US新闻数据文件: {news_raw_dir}")
            
            for news_file in news_files:
                output_file = news_processed_dir / news_file.name
                self.preprocess_news_data(str(news_file), str(output_file))
            
            self.logger.info("完成处理US新闻数据")
            
        except Exception as e:
            self.logger.error(f"处理US新闻数据失败: {str(e)}")
            raise

def main():
    """主函数"""
    try:
        # 设置数据根目录
        data_root = "Data"
        
        # 检查数据根目录是否存在
        if not os.path.exists(data_root):
            raise ValueError(f"数据根目录不存在: {data_root}")
        
        # 创建预处理器实例
        preprocessor = USNewsPreprocessor(data_root)
        
        # 处理US新闻数据
        preprocessor.process_directory()
        
        
    except Exception as e:
        logging.error(f"US新闻数据处理失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 