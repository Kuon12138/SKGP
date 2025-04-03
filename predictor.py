import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
import requests
import logging
from sklearn.metrics import accuracy_score, matthews_corrcoef
from config import ModelConfig
import pandas as pd


class Predictor:
    """预测器类"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._init_models()

    def _init_models(self):
        """初始化模型"""
        # 初始化BERT模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_model)
        self.model = AutoModel.from_pretrained(self.config.bert_model)
        
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
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.config.ollama_url, json=data)
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
        prompt = f"""你是一个专业的金融分析师。请从以下新闻中提取可能影响{stock_target}股价的前{self.config.k_factors}个因素：

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
            if len(stock_data) < self.config.window_size:
                raise ValueError(f"数据量不足，需要至少{self.config.window_size}天的数据")

            predictions = []
            actuals = []

            # 对每个交易日进行预测
            for i in range(self.config.window_size, len(stock_data)):
                try:
                    date_target = stock_data.index[i].strftime('%Y-%m-%d')
                    price_history = stock_data['Close'].iloc[i - self.config.window_size:i].tolist()
                    date_history = stock_data.index[i - self.config.window_size:i].strftime('%Y-%m-%d').tolist()

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