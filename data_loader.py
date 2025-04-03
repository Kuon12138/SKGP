import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
import logging
from config import ModelConfig


class DataLoader:
    """数据加载器"""
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.data_root = Path(model_config.data_root)
        self.news_path = self.data_root / "CMIN-US" / "news" / "preprocessed"
        self.price_path = self.data_root / "CMIN-US" / "price" / "preprocessed"
        self.market = "US"  # 默认市场
        self.logger = logging.getLogger(__name__)
        self._init_paths()

    def _init_paths(self):
        """初始化数据路径"""
        self.data_path = self.data_root / self.model_config.dataset
        self.price_path = self.data_path / 'price' / 'processed'
        self.news_path = self.data_path / 'news' / 'processed'
        self._validate_paths()

    def _validate_paths(self):
        """验证数据路径是否存在"""
        if not self.price_path.exists():
            raise FileNotFoundError(f"价格数据路径不存在: {self.price_path}")
        if not self.news_path.exists():
            raise FileNotFoundError(f"新闻数据路径不存在: {self.news_path}")

    def set_market(self, market: str):
        """设置市场类型"""
        self.market = market
        if market == "CN":
            self.news_path = self.data_root / "CMIN-CN" / "news" / "preprocessed"
            self.price_path = self.data_root / "CMIN-CN" / "price" / "preprocessed"
        else:
            self.news_path = self.data_root / "CMIN-US" / "news" / "preprocessed"
            self.price_path = self.data_root / "CMIN-US" / "price" / "preprocessed"

    def load_dataset(self, stock_symbol: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """加载数据集"""
        # 根据股票代码判断市场
        if stock_symbol.endswith(('.SZ', '.SH')):
            self.set_market("CN")
        else:
            self.set_market("US")
            
        stock_data = self.load_stock_data(stock_symbol, start_date, end_date)
        news_data = self.load_news_data(stock_symbol, start_date, end_date)
        return stock_data, news_data

    def load_stock_data(self, stock_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载股票数据

        Args:
            stock_symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            股票数据DataFrame
        """
        data_path = self.price_path / f"{stock_symbol}.txt"
        
        if not data_path.exists():
            raise FileNotFoundError(f"找不到股票数据文件: {data_path}")
        
        # 读取数据
        df = pd.read_csv(data_path)
        
        # 数据验证
        required_columns = ['Date', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据文件缺少必要的列: {missing_columns}")
        
        # 日期处理
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 数据清洗
        df = df.dropna(subset=['Close'])
        df = df.sort_index()
        
        # 过滤日期范围
        mask = (df.index >= start_date) & (df.index <= end_date)
        df = df[mask]
        
        if len(df) == 0:
            raise ValueError(f"在指定日期范围内没有数据: {start_date} 到 {end_date}")
        
        return df

    def load_news_data(self, stock_symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """
        加载新闻数据

        Args:
            stock_symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            新闻数据列表
        """
        data_path = self.news_path / f"{stock_symbol}.csv"
        
        if not data_path.exists():
            raise FileNotFoundError(f"找不到新闻数据文件: {data_path}")
        
        # 读取数据
        df = pd.read_csv(data_path)
        
        # 数据验证
        required_columns = ['publishedAt', 'title', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据文件缺少必要的列: {missing_columns}")
        
        # 日期处理
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        
        # 数据清洗
        df = df.dropna(subset=['title', 'description'])
        df = df.sort_values('publishedAt')
        
        # 过滤日期范围
        mask = (df['publishedAt'] >= start_date) & (df['publishedAt'] <= end_date)
        df = df[mask]
        
        if len(df) == 0:
            raise ValueError(f"在指定日期范围内没有新闻数据: {start_date} 到 {end_date}")
        
        # 转换为字典列表格式
        news_list = df.to_dict('records')
        
        return news_list 