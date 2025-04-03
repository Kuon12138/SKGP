import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List
import re

class DataPreprocessor:
    """数据预处理器"""
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

    def preprocess_stock_data(self, input_file: str, output_file: str) -> None:
        """
        预处理股票数据

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        try:
            # 读取数据
            df = pd.read_csv(input_file)
            
            # 数据清洗
            df = self._clean_stock_data(df)
            
            # 保存处理后的数据
            df.to_csv(output_file, index=False)
            self.logger.info(f"股票数据预处理完成: {output_file}")
            
        except Exception as e:
            self.logger.error(f"股票数据预处理失败: {str(e)}")
            raise

    def preprocess_news_data(self, input_file: str, output_file: str) -> None:
        """
        预处理新闻数据

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        try:
            # 读取数据
            df = pd.read_csv(input_file)
            
            # 数据清洗
            df = self._clean_news_data(df)
            
            # 保存处理后的数据
            df.to_csv(output_file, index=False)
            self.logger.info(f"新闻数据预处理完成: {output_file}")
            
        except Exception as e:
            self.logger.error(f"新闻数据预处理失败: {str(e)}")
            raise

    def _clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗股票数据"""
        # 确保日期格式正确
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 删除空值
        df = df.dropna(subset=['Close'])
        
        # 按日期排序
        df = df.sort_values('Date')
        
        # 去除异常值（使用3个标准差）
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df = df[abs(df[col] - mean) <= 3 * std]
        
        return df

    def _clean_news_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗新闻数据"""
        # 确保日期格式正确
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        
        # 删除空值
        df = df.dropna(subset=['title', 'description'])
        
        # 按时间排序
        df = df.sort_values('publishedAt')
        
        # 去除重复新闻
        df = df.drop_duplicates(subset=['title'])
        
        # 清理文本
        df['title'] = df['title'].apply(self._clean_text)
        df['description'] = df['description'].apply(self._clean_text)
        
        return df

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if not isinstance(text, str):
            return ""
        
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 去除特殊字符
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

def main():
    """主函数"""
    # 设置数据根目录
    data_root = "Data"
    preprocessor = DataPreprocessor(data_root)
    
    # 处理US数据
    us_stock_dir = Path(data_root) / "CMIN-US" / "price" / "raw"
    us_news_dir = Path(data_root) / "CMIN-US" / "news" / "raw"
    
    # 处理CN数据
    cn_stock_dir = Path(data_root) / "CMIN-CN" / "price" / "raw"
    cn_news_dir = Path(data_root) / "CMIN-CN" / "news" / "raw"
    
    # 处理所有数据文件
    for market in ['US', 'CN']:
        stock_dir = us_stock_dir if market == 'US' else cn_stock_dir
        news_dir = us_news_dir if market == 'US' else cn_news_dir
        
        # 创建输出目录
        stock_output_dir = stock_dir.parent / "processed"
        news_output_dir = news_dir.parent / "processed"
        stock_output_dir.mkdir(exist_ok=True)
        news_output_dir.mkdir(exist_ok=True)
        
        # 处理股票数据
        for stock_file in stock_dir.glob("*.csv"):
            output_file = stock_output_dir / stock_file.name
            preprocessor.preprocess_stock_data(str(stock_file), str(output_file))
        
        # 处理新闻数据
        for news_file in news_dir.glob("*.csv"):
            output_file = news_output_dir / news_file.name
            preprocessor.preprocess_news_data(str(news_file), str(output_file))

if __name__ == "__main__":
    main() 