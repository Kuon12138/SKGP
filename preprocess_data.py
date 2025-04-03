import logging
import os
import pandas as pd
from pathlib import Path
from typing import Optional

class DataPreprocessor:
    def __init__(self, data_root: str):
        """初始化数据预处理器
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

    def preprocess_price_data(self, input_file: str, output_file: str) -> None:
        """处理价格数据
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(input_file)
            
            # 将列名转换为小写
            df.columns = [col.lower().strip() for col in df.columns]
            
            # 定义可能的列名映射
            date_columns = ['date', 'time', 'datetime', '日期', 'trade_date', 'trading_date']
            open_columns = ['open', 'opening', '开盘价', 'open_price']
            high_columns = ['high', 'highest', '最高价', 'high_price']
            low_columns = ['low', 'lowest', '最低价', 'low_price']
            close_columns = ['close', 'closing', '收盘价', 'close_price']
            volume_columns = ['volume', 'vol', '成交量', 'trade_volume']
            
            # 查找实际的列名
            def find_column(possible_names, df_columns):
                for name in possible_names:
                    if name in df_columns:
                        return name
                return None
            
            # 获取实际的列名
            date_col = find_column(date_columns, df.columns)
            open_col = find_column(open_columns, df.columns)
            high_col = find_column(high_columns, df.columns)
            low_col = find_column(low_columns, df.columns)
            close_col = find_column(close_columns, df.columns)
            volume_col = find_column(volume_columns, df.columns)
            
            # 检查是否找到所有必要的列
            if not all([date_col, open_col, high_col, low_col, close_col, volume_col]):
                missing = []
                if not date_col: missing.append('date')
                if not open_col: missing.append('open')
                if not high_col: missing.append('high')
                if not low_col: missing.append('low')
                if not close_col: missing.append('close')
                if not volume_col: missing.append('volume')
                raise ValueError(f"价格数据缺少必要列 {missing}: {input_file}")
            
            # 创建新的数据框并标准化列名
            new_df = pd.DataFrame()
            new_df['date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
            new_df['open'] = pd.to_numeric(df[open_col], errors='coerce')
            new_df['high'] = pd.to_numeric(df[high_col], errors='coerce')
            new_df['low'] = pd.to_numeric(df[low_col], errors='coerce')
            new_df['close'] = pd.to_numeric(df[close_col], errors='coerce')
            new_df['volume'] = pd.to_numeric(df[volume_col], errors='coerce')
            
            # 处理异常值
            new_df = new_df[
                (new_df['open'] > 0) &
                (new_df['high'] > 0) &
                (new_df['low'] > 0) &
                (new_df['close'] > 0) &
                (new_df['volume'] >= 0)
            ]
            
            # 删除缺失值
            new_df = new_df.dropna()
            
            # 按日期排序并删除重复行
            new_df = new_df.sort_values('date').drop_duplicates()
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存处理后的数据
            new_df.to_csv(output_file, index=False)
            self.logger.info(f"价格数据处理完成: {output_file}")
            
        except Exception as e:
            self.logger.error(f"价格数据处理失败: {str(e)}")
            self.logger.error(f"文件路径: {input_file}")
            raise

    def preprocess_news_data(self, input_file: str, output_file: str) -> None:
        """处理新闻数据
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        try:
            # 尝试读取CSV文件，处理不一致的列数
            try:
                df = pd.read_csv(input_file)
            except pd.errors.ParserError:
                # 如果出现解析错误，尝试使用更宽松的解析方式
                df = pd.read_csv(input_file, engine='python', on_bad_lines='skip')
            
            if df.empty:
                raise ValueError(f"空数据文件: {input_file}")
            
            # 重置索引以避免重复标签问题
            df = df.reset_index(drop=True)
            
            # 标准化列名
            df.columns = [col.lower().strip() for col in df.columns]
            
            # 创建新的数据框
            new_df = pd.DataFrame(index=df.index)
            
            # 处理日期列
            date_patterns = ['date', 'time', 'publishedat', '日期']
            date_col = None
            
            # 查找日期列
            for pattern in date_patterns:
                matching_cols = [col for col in df.columns if pattern in col.lower()]
                if matching_cols:
                    date_col = matching_cols[0]
                    break
            
            # 如果没找到日期列，使用第一列
            if date_col is None:
                date_col = df.columns[0]
            
            # 改进的日期处理
            def clean_date(date_str):
                try:
                    # 如果日期包含时间，只取日期部分
                    if isinstance(date_str, str) and ' ' in date_str:
                        date_str = date_str.split()[0]
                    return pd.to_datetime(date_str, format='mixed').strftime('%Y-%m-%d')
                except:
                    try:
                        return pd.to_datetime(date_str).strftime('%Y-%m-%d')
                    except:
                        return None
            
            # 处理日期格式
            new_df['publishedAt'] = df[date_col].apply(clean_date)
            
            # 删除无效日期的行
            new_df = new_df[new_df['publishedAt'].notna()].copy()
            
            # 处理标题列
            title_patterns = ['title', 'headline', 'news', '标题']
            title_col = None
            
            # 查找标题列
            for pattern in title_patterns:
                matching_cols = [col for col in df.columns if pattern in col.lower()]
                if matching_cols:
                    title_col = matching_cols[0]
                    break
            
            # 如果没找到标题列，使用第二列
            if title_col is None and len(df.columns) > 1:
                title_col = df.columns[1]
            elif title_col is None:
                raise ValueError(f"无法找到标题列: {input_file}")
            
            # 重置索引以确保一致性
            df = df.loc[new_df.index].reset_index(drop=True)
            new_df = new_df.reset_index(drop=True)
            
            # 添加标题和描述
            new_df['title'] = df[title_col].astype(str).str.strip()
            new_df['description'] = new_df['title']
            
            # 按日期排序并删除重复行
            new_df = new_df.sort_values('publishedAt').reset_index(drop=True)
            new_df = new_df.drop_duplicates().reset_index(drop=True)
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存处理后的数据
            new_df.to_csv(output_file, index=False)
            self.logger.info(f"新闻数据处理完成: {output_file}")
            
        except Exception as e:
            self.logger.error(f"新闻数据处理失败: {str(e)}")
            self.logger.error(f"文件路径: {input_file}")
            raise

    def process_directory(self, market: str) -> None:
        """处理指定市场的所有数据文件
        Args:
            market: 市场类型（'CN' 或 'US'）
        """
        try:
            # 构建目录路径
            market_dir = self.data_root / f"CMIN-{market}"
            
            # 定义原始数据和处理后数据的目录
            price_raw_dir = market_dir / "price" / "raw"
            news_raw_dir = market_dir / "news" / "raw"
            price_processed_dir = market_dir / "price" / "processed"
            news_processed_dir = market_dir / "news" / "processed"
            
            # 创建处理后数据的目录
            os.makedirs(price_processed_dir, exist_ok=True)
            os.makedirs(news_processed_dir, exist_ok=True)
            
            # 检查原始数据目录是否存在
            if not price_raw_dir.exists():
                raise ValueError(f"价格数据目录不存在: {price_raw_dir}")
            if not news_raw_dir.exists():
                raise ValueError(f"新闻数据目录不存在: {news_raw_dir}")
            
            # 处理价格数据
            price_files = list(price_raw_dir.glob("*.csv"))
            if not price_files:
                self.logger.warning(f"没有找到价格数据文件: {price_raw_dir}")
            for price_file in price_files:
                output_file = price_processed_dir / price_file.name
                self.preprocess_price_data(str(price_file), str(output_file))
            
            # 处理新闻数据
            news_files = list(news_raw_dir.glob("*.csv"))
            if not news_files:
                self.logger.warning(f"没有找到新闻数据文件: {news_raw_dir}")
            for news_file in news_files:
                output_file = news_processed_dir / news_file.name
                self.preprocess_news_data(str(news_file), str(output_file))
            
            self.logger.info(f"完成处理{market}市场数据")
            
        except Exception as e:
            self.logger.error(f"处理{market}市场数据失败: {str(e)}")
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
        preprocessor = DataPreprocessor(data_root)
        
        # 处理中国和美国市场数据
        for market in ["CN", "US"]:
            try:
                preprocessor.process_directory(market)
            except Exception as e:
                logging.error(f"处理{market}市场数据时出错: {str(e)}")
                continue
        
        logging.info("数据处理完成")
        
    except Exception as e:
        logging.error(f"数据处理失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()