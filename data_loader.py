import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import yaml

class DataLoader:
    def __init__(self, config_path: str = "config.yaml"):
        """初始化数据加载器
        Args:
            config_path: 配置文件路径
        """
        self._setup_logging()
        self.config = self._load_config(config_path)
        self.data_root = Path(self.config['model']['data_root'])

    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件
        Args:
            config_path: 配置文件路径
        Returns:
            配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            raise

    def load_data(self, market: str, symbol: str, 
                  start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载指定市场和股票的数据
        Args:
            market: 市场类型（'CN' 或 'US'）
            symbol: 股票代码
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
        Returns:
            价格数据和新闻数据的元组
        """
        try:
            # 构建文件路径
            market_dir = self.data_root / f"CMIN-{market}"
            price_file = market_dir / "price" / "processed" / f"{symbol}.csv"
            news_file = market_dir / "news" / "processed" / f"{symbol}.csv"

            # 检查文件是否存在
            if not price_file.exists():
                raise FileNotFoundError(f"价格数据文件不存在: {price_file}")
            if not news_file.exists():
                raise FileNotFoundError(f"新闻数据文件不存在: {news_file}")

            # 加载数据
            price_df = pd.read_csv(price_file)
            news_df = pd.read_csv(news_file)

            # 确保日期列格式正确
            price_df['date'] = pd.to_datetime(price_df['date'])
            news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])

            # 应用日期过滤
            if start_date:
                start_date = pd.to_datetime(start_date)
                price_df = price_df[price_df['date'] >= start_date]
                news_df = news_df[news_df['publishedAt'] >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                price_df = price_df[price_df['date'] <= end_date]
                news_df = news_df[news_df['publishedAt'] <= end_date]

            # 按日期排序
            price_df = price_df.sort_values('date')
            news_df = news_df.sort_values('publishedAt')

            # 重置索引
            price_df = price_df.reset_index(drop=True)
            news_df = news_df.reset_index(drop=True)

            return price_df, news_df

        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            raise

    def get_latest_data(self, market: str, symbol: str, 
                       window_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """获取最新的数据窗口
        Args:
            market: 市场类型（'CN' 或 'US'）
            symbol: 股票代码
            window_size: 窗口大小
        Returns:
            最新的价格数据和新闻数据的元组
        """
        try:
            # 加载所有数据
            price_df, news_df = self.load_data(market, symbol)

            # 获取最新的价格数据窗口
            latest_price = price_df.tail(window_size)

            # 获取对应时间段的新闻数据
            start_date = latest_price['date'].iloc[0]
            latest_news = news_df[news_df['publishedAt'] >= start_date]

            return latest_price, latest_news

        except Exception as e:
            self.logger.error(f"获取最新数据失败: {str(e)}")
            raise

    def prepare_features(self, price_df: pd.DataFrame, 
                    news_df: pd.DataFrame, 
                    window_size: int) -> Tuple[pd.DataFrame, pd.Series]:
        """准备模型特征
        Args:
            price_df: 价格数据
            news_df: 新闻数据
            window_size: 窗口大小
        Returns:
            特征数据框和标签序列的元组
        """
        try:
            # 确保价格数据有足够的历史数据
            if len(price_df) < window_size:
                raise ValueError(f"价格数据不足，需要至少 {window_size} 条数据，但只有 {len(price_df)} 条")
            
            self.logger.info(f"开始准备特征，价格数据长度: {len(price_df)}")
            
            # 创建特征数据框
            features = pd.DataFrame(index=price_df.index)
            
            # 基本价格特征
            features['price_change'] = price_df['close'].pct_change()
            features['volume_change'] = price_df['volume'].pct_change()
            
            # 使用较小的移动窗口
            ma_windows = [3, 5, 10]  # 减小移动平均窗口大小
            for window in ma_windows:
                features[f'ma{window}'] = price_df['close'].rolling(window, min_periods=1).mean()
            
            # 计算较短期的波动率
            vol_window = min(5, window_size)  # 使用较小的波动率窗口
            features['volatility'] = price_df['close'].rolling(vol_window, min_periods=1).std()
            
            # 计算RSI（使用较短的周期）
            rsi_period = 7  # 减小RSI周期
            delta = price_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            features['rsi'] = features['rsi'].fillna(50)  # 填充RSI的NaN值
            
            # 添加新闻数量特征
            news_counts = pd.Series(0, index=price_df.index)
            for date in price_df.index:
                date_str = pd.to_datetime(price_df.loc[date, 'date']).strftime('%Y-%m-%d')
                news_counts[date] = len(news_df[
                    pd.to_datetime(news_df['publishedAt']).dt.strftime('%Y-%m-%d') == date_str
                ])
            features['news_count'] = news_counts
            
            # 填充缺失值
            features = features.fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info(f"特征计算完成，特征数量: {len(features.columns)}")
            self.logger.info(f"特征列名: {features.columns.tolist()}")
            
            if features.empty:
                raise ValueError("特征生成后数据为空")
            
            if features.isna().any().any():
                self.logger.warning("特征中仍然存在NaN值")
                self.logger.warning(f"NaN值统计:\n{features.isna().sum()}")
            
            # 计算标签（1表示上涨，0表示下跌或持平）
            labels = (price_df['close'].shift(-1) > price_df['close']).astype(int)
            labels = labels[features.index]
            
            return features, labels

        except Exception as e:
            self.logger.error(f"准备特征失败: {str(e)}")
            self.logger.error(f"价格数据形状: {price_df.shape}")
            self.logger.error(f"新闻数据形状: {news_df.shape}")
            raise

def main():
    """测试数据加载器"""
    try:
        # 创建数据加载器实例
        loader = DataLoader()
        
        # 测试加载中国市场数据
        cn_price, cn_news = loader.load_data(
            market="CN",
            symbol=loader.config['data']['cn_stock'],
            start_date=loader.config['data']['start_date'],
            end_date=loader.config['data']['end_date']
        )
        print("成功加载中国市场数据")
        print(f"价格数据形状: {cn_price.shape}")
        print(f"新闻数据形状: {cn_news.shape}")
        
        # 测试加载美国市场数据
        us_price, us_news = loader.load_data(
            market="US",
            symbol=loader.config['data']['us_stock'],
            start_date=loader.config['data']['start_date'],
            end_date=loader.config['data']['end_date']
        )
        print("成功加载美国市场数据")
        print(f"价格数据形状: {us_price.shape}")
        print(f"新闻数据形状: {us_news.shape}")
        
    except Exception as e:
        logging.error(f"测试数据加载器失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()