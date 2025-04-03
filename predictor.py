import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
import requests
import logging
from sklearn.metrics import accuracy_score, matthews_corrcoef
from config import ModelConfig
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from data_loader import DataLoader
import json
import os


class Predictor:
    """预测器类"""
    def __init__(self, config_path: str = "config.yaml"):
        """初始化预测器
        Args:
            config_path: 配置文件路径
        """
        self._setup_logging()
        self.config = self._load_config(config_path)
        self.data_loader = DataLoader(config_path)
        self._init_models()

    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            raise

    def _init_models(self):
        """初始化模型"""
        # 初始化BERT模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['bert_model'])
        self.model = AutoModel.from_pretrained(self.config['model']['bert_model'])
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _call_ollama(self, prompt: str) -> str:
        """
        调用Ollama API

        Args:
            prompt: 提示文本

        Returns:
            API响应文本
        """
        data = {
            "model": self.config['model']['ollama_model'],
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.config['model']['ollama_url'], json=data)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            self.logger.error(f"调用Ollama API时出错: {e}")
            return ""

    def get_stock_movement(self, prices: List[float]) -> List[int]:
        """
        将价格序列转换为涨跌序列

        Args:
            prices: 价格列表

        Returns:
            涨跌序列(1表示上涨，0表示下跌)
        """
        movements = []
        for i in range(1, len(prices)):
            movements.append(1 if prices[i] > prices[i - 1] else 0)
        return movements

    def get_relation(self, stock_target: str, news_text: str) -> str:
        """
        获取目标股票与新闻中提到的其他股票之间的关系

        Args:
            stock_target: 目标股票代码
            news_text: 新闻文本

        Returns:
            关系描述
        """
        prompt = f"""你是一个专业的金融分析师。请分析以下新闻内容，并填空：
{stock_target}和新闻中提到的其他股票最有可能处于___关系

新闻内容：
{news_text}

请只填写关系类型，不要添加其他解释。"""

        return self._call_ollama(prompt)

    def extract_factors(self, stock_target: str, news_text: str) -> List[str]:
        """
        从新闻中提取可能影响股价的因素

        Args:
            stock_target: 目标股票代码
            news_text: 新闻文本

        Returns:
            提取的因素列表
        """
        prompt = f"""你是一个专业的金融分析师。请从以下新闻中提取可能影响{stock_target}股价的前{self.config['model']['k_factors']}个因素：

新闻内容：
{news_text}

请列出因素，每行一个，不要添加编号或其他解释。"""

        response = self._call_ollama(prompt)
        factors = response.split('\n')
        return [f.strip() for f in factors if f.strip()]

    def predict_stock_movement(self,
                             stock_target: str,
                             date_target: str,
                             news_target: str,
                             price_history: List[float],
                             date_history: List[str]) -> Tuple[int, str]:
        """
        预测股票走势

        Args:
            stock_target: 目标股票代码
            date_target: 目标日期
            news_target: 目标新闻
            price_history: 历史价格列表
            date_history: 历史日期列表

        Returns:
            预测结果(1表示上涨，0表示下跌)和推理理由
        """
        try:
            # 输入验证
            if not price_history or not date_history:
                raise ValueError("历史价格数据为空")
            if not news_target:
                raise ValueError("新闻数据为空")
            if len(price_history) != len(date_history):
                raise ValueError("历史价格和日期数据长度不匹配")

            # 获取股票走势序列
            movements = self.get_stock_movement(price_history)
            text_movements = ["上涨" if m == 1 else "下跌" for m in movements]

            # 构建时间模板
            time_template = "\n".join([
                f"在{date_history[i]}，{stock_target}的股价{text_movements[i]}"
                for i in range(len(movements))
            ])

            # 获取关系信息
            relation = self.get_relation(stock_target, news_target)
            if not relation:
                raise ValueError("无法获取股票关系信息")

            # 提取因素
            factors = self.extract_factors(stock_target, news_target)
            if not factors:
                raise ValueError("无法提取影响因素")

            # 构建预测提示
            prompt = f"""你是一个专业的金融分析师。请根据以下信息，判断股价的走势是上涨还是下跌，填写空白并给出理由：

关系：{relation}

因素：
{chr(10).join(factors)}

历史走势：
{time_template}

在{date_target}，{stock_target}的股价将___。

请先填写"上涨"或"下跌"，然后给出详细的分析理由。"""

            # 获取预测结果
            prediction_text = self._call_ollama(prompt)
            if not prediction_text:
                raise ValueError("无法获取预测结果")

            # 解析预测结果
            prediction = 1 if "上涨" in prediction_text else 0

            return prediction, prediction_text

        except Exception as e:
            self.logger.error(f"预测过程中出错: {str(e)}")
            raise

    def evaluate(self,
                 stock_data: pd.DataFrame,
                 news_data: List[Dict],
                 stock_symbol: str) -> Dict[str, float]:
        """
        评估模型性能

        Args:
            stock_data: 股票数据
            news_data: 新闻数据
            stock_symbol: 股票代码

        Returns:
            包含准确率和MCC的评估结果
        """
        try:
            if len(stock_data) < self.config['model']['window_size']:
                raise ValueError(f"数据量不足，需要至少{self.config['model']['window_size']}天的数据")

            predictions = []
            actuals = []

            # 对每个交易日进行预测
            for i in range(self.config['model']['window_size'], len(stock_data)):
                try:
                    date_target = stock_data.index[i].strftime('%Y-%m-%d')
                    price_history = stock_data['Close'].iloc[i - self.config['model']['window_size']:i].tolist()
                    date_history = stock_data.index[i - self.config['model']['window_size']:i].strftime('%Y-%m-%d').tolist()

                    # 获取当天的新闻
                    day_news = [n for n in news_data if n['publishedAt'].strftime('%Y-%m-%d') == date_target]
                    if not day_news:
                        self.logger.warning(f"警告：{date_target}没有相关新闻，跳过该日期")
                        continue

                    news_text = "\n".join([n['title'] + "\n" + n['description'] for n in day_news])

                    # 进行预测
                    prediction, _ = self.predict_stock_movement(
                        stock_symbol,
                        date_target,
                        news_text,
                        price_history,
                        date_history
                    )

                    predictions.append(prediction)
                    actuals.append(1 if stock_data['Close'].iloc[i] > stock_data['Close'].iloc[i - 1] else 0)

                except Exception as e:
                    self.logger.error(f"处理{date_target}数据时出错: {str(e)}")
                    continue

            if not predictions or not actuals:
                raise ValueError("没有成功完成任何预测")

            # 计算评估指标
            acc = accuracy_score(actuals, predictions)
            mcc = matthews_corrcoef(actuals, predictions)

            return {
                'accuracy': acc,
                'mcc': mcc,
                'total_predictions': len(predictions),
                'successful_predictions': sum(1 for p, a in zip(predictions, actuals) if p == a)
            }

        except Exception as e:
            self.logger.error(f"评估过程中出错: {str(e)}")
            raise

    def predict(self, market: str, symbol: str) -> Dict:
        """进行预测
        Args:
            market: 市场类型（'CN' 或 'US'）
            symbol: 股票代码
        Returns:
            预测结果字典
        """
        try:
            # 获取最新数据
            window_size = self.config['model']['window_size']
            price_df, news_df = self.data_loader.get_latest_data(
                market, symbol, window_size
            )
            
            # 检查数据是否足够
            if len(price_df) < window_size:
                raise ValueError(f"价格数据不足，需要至少 {window_size} 条数据，但只有 {len(price_df)} 条")
            
            # 准备特征
            features, _ = self.data_loader.prepare_features(
                price_df, news_df, window_size
            )
            
            # 检查特征是否为空
            if features.empty:
                raise ValueError("无法生成特征数据")
            
            # 获取最新的特征数据
            latest_features = features.iloc[-1:]
            
            # 获取最新的新闻数据
            latest_date = price_df['date'].max()
            latest_news = news_df[
                pd.to_datetime(news_df['publishedAt']).dt.date == 
                pd.to_datetime(latest_date).date()
            ]
            
            # 调用LLM进行分析
            prediction, reasoning = self._analyze_with_llm(
                latest_features, latest_news, market, symbol
            )
            
            # 将Timestamp转换为字符串
            latest_date_str = latest_date.strftime('%Y-%m-%d') if latest_date else None
            
            # 处理特征数据，确保所有值都是JSON可序列化的
            feature_dict = {}
            for key, value in latest_features.to_dict('records')[0].items():
                if isinstance(value, (pd.Timestamp, np.datetime64)):
                    feature_dict[key] = value.strftime('%Y-%m-%d')
                elif isinstance(value, (np.int64, np.int32)):
                    feature_dict[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    feature_dict[key] = float(value)
                else:
                    feature_dict[key] = value
            
            return {
                'date': latest_date_str,
                'prediction': int(prediction),  # 确保是普通的int类型
                'confidence': float(self._calculate_confidence(latest_features)),  # 确保是普通的float类型
                'reasoning': str(reasoning),  # 确保是普通的字符串
                'features': feature_dict,
                'news_count': int(len(latest_news))  # 确保是普通的int类型
            }
            
        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            return {
                'date': None,
                'prediction': 0,
                'confidence': 0.0,
                'reasoning': f"预测失败: {str(e)}",
                'features': {},
                'news_count': 0
            }

    def _analyze_with_llm(self, features: pd.DataFrame, 
                         news_df: pd.DataFrame, 
                         market: str, 
                         symbol: str) -> Tuple[int, str]:
        """使用LLM分析数据并生成预测
        Returns:
            (预测结果, 推理过程) 的元组
        """
        try:
            # 准备提示信息
            prompt = self._prepare_prompt(features, news_df, market, symbol)
            
            # 调用Ollama API
            response = requests.post(
                self.config['model']['ollama_url'],
                json={
                    "model": self.config['model']['ollama_model'],
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API调用失败: {response.text}")
            
            # 解析响应
            result = response.json()
            analysis = result['response']
            
            # 解析预测结果
            prediction = 1 if "上涨" in analysis else 0
            
            return prediction, analysis
            
        except Exception as e:
            self.logger.error(f"LLM分析失败: {str(e)}")
            raise

    def _prepare_prompt(self, features: pd.DataFrame, 
                       news_df: pd.DataFrame, 
                       market: str, 
                       symbol: str) -> str:
        """准备LLM提示信息"""
        try:
            # 检查特征数据
            if features.empty:
                raise ValueError("特征数据为空")
            
            # 获取特征值
            feature_values = features.iloc[0]
            
            prompt = f"""分析以下数据，预测{market}市场{symbol}股票明天的走势：

技术指标：
- 价格变动：{feature_values['price_change']:.2%}
- 成交量变动：{feature_values['volume_change']:.2%}
- RSI：{feature_values['rsi']:.2f}
- 波动率：{feature_values['volatility']:.2f}
- 3日均线：{feature_values['ma3']:.2f}
- 5日均线：{feature_values['ma5']:.2f}
- 10日均线：{feature_values['ma10']:.2f}
- 今日新闻数量：{feature_values['news_count']}

今日新闻：
"""
            # 添加新闻内容
            if len(news_df) > 0:
                for _, news in news_df.iterrows():
                    prompt += f"- {news['title']}\n"
            else:
                prompt += "（今日无相关新闻）\n"
            
            prompt += "\n请分析这些数据，并给出明天的预测（上涨或下跌）及理由。"
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"准备提示信息失败: {str(e)}")
            raise

    def _calculate_confidence(self, features: pd.DataFrame) -> float:
        """计算预测置信度"""
        # 这里可以实现更复杂的置信度计算逻辑
        return 0.7

def save_prediction(result: Dict, stock_code: str, result_dir: str = "result"):
    """保存预测结果到文件
    Args:
        result: 预测结果字典
        stock_code: 股票代码
        result_dir: 结果保存目录
    """
    try:
        # 创建结果目录
        os.makedirs(result_dir, exist_ok=True)
        
        # 构建文件路径
        file_path = os.path.join(result_dir, f"{stock_code}.json")
        
        # 保存结果
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        logging.info(f"预测结果已保存到: {file_path}")
        
    except Exception as e:
        logging.error(f"保存预测结果失败: {str(e)}")

def main():
    """测试预测器"""
    try:
        predictor = Predictor()
        
        # 获取要预测的股票列表
        stocks = predictor.config['data'].get('stocks', {
            'CN': [predictor.config['data'].get('cn_stock', '600519.SH')],
            'US': [predictor.config['data'].get('us_stock', 'AAPL')]
        })
        
        # 测试中国市场预测
        print("\n中国市场预测结果:")
        for stock in stocks['CN']:
            try:
                result = predictor.predict("CN", stock)
                print(f"\n{stock} 预测结果:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                # 保存预测结果
                save_prediction(result, stock)
            except Exception as e:
                logging.error(f"预测 {stock} 失败: {str(e)}")
                continue
        
        # 测试美国市场预测
        print("\n美国市场预测结果:")
        for stock in stocks['US']:
            try:
                result = predictor.predict("US", stock)
                print(f"\n{stock} 预测结果:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                # 保存预测结果
                save_prediction(result, stock)
            except Exception as e:
                logging.error(f"预测 {stock} 失败: {str(e)}")
                continue
        
    except Exception as e:
        logging.error(f"测试预测器失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 