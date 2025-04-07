import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
import logging
from sklearn.metrics import accuracy_score, matthews_corrcoef
import pandas as pd
import yaml
from data_loader_test import DataLoader
import json
import os
from datetime import datetime
from openai import OpenAI as openai


class Predictor:
    """预测器类"""

    """预测器类"""

    def __init__(self, config_path: str = "configtest.yaml"):
        """初始化预测器
        Args:
            config_path: 配置文件路径
        """

        self._setup_logging()
        self.config = self._load_config(config_path)
        openai.api_key = self.config['openai']['api_key']

        # GPU设置和性能测试
        gpu_available = self._setup_gpu()

        if not gpu_available:
            # 如果GPU不可用，调整配置
            self.logger.info("切换到CPU模式，调整配置...")
            self.config['model']['gpu_settings'].update({
                'num_gpu_layers': 0,
                'batch_size': 32,  # 减小批处理大小
                'num_threads': max(1, os.cpu_count() - 1)  # 使用CPU线程
            })


        self.data_loader = DataLoader(config_path)
        self._init_models()
        self._response_cache = {}  # 添加响应缓存


    def _setup_logging(self):
        """设置日志配置"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False  # 防止日志向上传递

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            raise

    def _setup_gpu(self):
        """优化GPU设置"""
        if torch.cuda.is_available():
            try:
                # 基础设置
                self.device = torch.device('cuda:0')

                # 设置GPU内存分配策略
                torch.cuda.set_per_process_memory_fraction(
                    self.config['model']['gpu_settings']['memory_fraction']
                )

                # 启用自动混合精度
                self.scaler = torch.amp.GradScaler('cuda')  # 修复弃用警告
                self.autocast = torch.amp.autocast('cuda')

                # 性能优化
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True

                # 静默执行GPU性能测试
                size = 2000
                a = torch.randn(size, size, device='cuda')
                b = torch.randn(size, size, device='cuda')
                torch.cuda.synchronize()
                _ = torch.mm(a, b)
                torch.cuda.synchronize()

                # 清理内存
                del a, b
                torch.cuda.empty_cache()

                return True

            except Exception as e:
                self.logger.error(f"GPU设置失败: {str(e)}")
                self.device = torch.device('cpu')
                return False

        self.device = torch.device('cpu')
        return False



    def _init_models(self):
        """初始化模型"""
        try:
            # 初始化BERT模型和tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['bert_model'])
            self.bert_model = AutoModel.from_pretrained(self.config['model']['bert_model'])

            # 将模型移动到正确的设备
            self.bert_model = self.bert_model.to(self.device)

            # 设置为评估模式
            self.bert_model.eval()

        except Exception as e:
            self.logger.error(f"BERT模型初始化失败: {str(e)}")
            raise

    def _get_openai_response(self, prompt: str) -> str:
        """获取openai API响应"""
        try:
            # 获取配置参数
            timeout = self.config['model']['timeout']
            max_prompt_length = self.config['model']['max_prompt_length']
            openai_model = self.config['model']['openai_model']

            # 获取生成参数
            generation_settings = self.config['model']['gpu_settings']['generation_settings']
            temperature = generation_settings['temperature']
            top_p = generation_settings['top_p']
            top_k = generation_settings['top_k']
            num_predict = generation_settings['num_predict']



            # 如果提示文本超过最大长度，进行智能截断
            if len(prompt) > max_prompt_length:
                self.logger.warning(f"提示文本过长 ({len(prompt)} > {max_prompt_length})，进行智能截断")

                # 1. 首先尝试保留关键部分
                important_parts = [
                    "关系：",
                    "因素：",
                    "历史走势：",
                    "预测："
                ]

                # 2. 查找关键部分的位置
                positions = []
                for part in important_parts:
                    pos = prompt.find(part)
                    if pos != -1:
                        positions.append((pos, part))

                # 3. 按位置排序
                positions.sort()

                # 4. 构建新的提示文本
                new_prompt = ""
                for pos, part in positions:
                    end_pos = prompt.find("\n\n", pos)
                    if end_pos == -1:
                        end_pos = len(prompt)
                    part_text = prompt[pos:end_pos]
                    if len(new_prompt) + len(part_text) <= max_prompt_length:
                        new_prompt += part_text + "\n\n"
                    else:
                        break

                # 5. 如果还是太长，进行简单截断
                if len(new_prompt) > max_prompt_length:
                    new_prompt = prompt[:max_prompt_length - 3] + "..."

                prompt = new_prompt

            client = openai(
                base_url="https://openrouter.ai/api/v1",
                api_key="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",
            )


            # 使用 openai.Completion.create() 方法代替 requests
            response = client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct:free",
                # 准备消息格式
                temperature=temperature,
                max_tokens=num_predict,  # OpenAI使用max_tokens限制生成长度
                top_p=top_p,
                timeout=timeout,
                messages=[
                    {"role": "user", "content": f"[INST] {prompt} [/INST]"}
                ]


            )


            response_text = response.choices[0].message.content.strip()
            return response_text




        except Exception as e:
            self.logger.error(f"openai API调用失败: {str(e)}")
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
        prompt = f"""你是一个专业的金融分析师。请分析以下新闻内容，并填空：{stock_target}和新闻中提到的其他股票最有可能处于___关系
        新闻内容：
{news_text}
请只填写关系类型，不要添加其他解释。"""

        return self._get_openai_response(prompt)

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

        response = self._get_openai_response(prompt)
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
关系：
{relation}

因素：
{chr(10).join(factors)}

历史走势：
{time_template}

在{date_target}，{stock_target}的股价将___。

请先填写"上涨"或"下跌"，然后给出详细的分析理由。"""

            # 获取预测结果
            prediction_text = self._get_openai_response(prompt)
            if not prediction_text:
                raise ValueError("无法获取预测结果")

            # 解析预测结果
            prediction = 1 if "上涨" in prediction_text else 0

            return prediction, prediction_text

        except Exception as e:
            self.logger.error(f"预测过程中出错: {str(e)}")
            raise

    def evaluate(self, stock_data: pd.DataFrame, news_data: pd.DataFrame, stock_symbol: str) -> Dict:
        """评估模型性能
        Args:
            stock_data: 股票数据
            news_data: 新闻数据
            stock_symbol: 股票代码
        Returns:
            评估结果字典
        """
        predictions = []
        actual_movements = []
        results = []

        for date in stock_data.index[:-1]:  # 除去最后一天
            try:
                next_day = stock_data.index[stock_data.index.get_loc(date) + 1]

                # 获取当天的预测
                result = self.predict_single_day(
                    stock_symbol,
                    date,
                    news_data[news_data.index <= date],
                    stock_data[:date],
                    stock_data.index
                )

                # 获取实际走势
                actual = 1 if stock_data.loc[next_day, 'close'] > stock_data.loc[date, 'close'] else 0

                predictions.append(result['prediction'])
                actual_movements.append(actual)
                results.append(result)

            except Exception as e:
                self.logger.error(f"评估日期 {date} 时发生错误: {str(e)}")

        # 计算评估指标
        accuracy = accuracy_score(actual_movements, predictions) if predictions else 0
        mcc = matthews_corrcoef(actual_movements, predictions) if predictions else 0

        return {
            'accuracy': accuracy,
            'mcc': mcc,
            'total_predictions': len(predictions),
            'results': results
        }

    def predict_single_day(self, stock: str, date: datetime,
                           news_df: pd.DataFrame, price_df: pd.DataFrame,
                           date_history: pd.DatetimeIndex) -> Dict:
        """单日预测
        Args:
            stock: 股票代码
            date: 预测日期
            news_df: 新闻数据
            price_df: 价格数据
            date_history: 日期历史
        Returns:
            预测结果字典
        """
        try:
            # 1. 获取背景知识
            knowledge = self._get_background_knowledge(stock, news_df)

            # 2. 提取影响因素
            factors = self._extract_factors(stock, news_df)

            # 3. 获取BERT特征
            bert_features = self._get_bert_features(news_df)

            # 4. 准备价格历史文本
            price_history = self._prepare_price_history(price_df)

            # 5. 构建最终预测提示
            prompt = self._build_prediction_prompt(
                price_data=price_df,
                news_data=news_df,
                bert_features=bert_features
            )

            # 6. 获取预测结果
            prediction, reasoning = self._get_prediction(prompt)

            return {
                'date': date.strftime('%Y-%m-%d'),
                'prediction': prediction,
                'confidence': self._calculate_confidence(factors, bert_features),
                'reasoning': reasoning,
                'knowledge': knowledge,
                'factors': factors
            }

        except Exception as e:
            self.logger.error(f"单日预测失败: {str(e)}")
            return self._get_default_prediction(date)

    def _get_bert_features(self, news_data: pd.DataFrame) -> torch.Tensor:
        """提取新闻文本的BERT特征"""
        try:
            if news_data.empty:
                return None

            # 获取配置
            bert_config = self.config['model']['bert_settings']
            batch_size = bert_config['batch_size']

            # 只使用title列
            texts = news_data['title'].fillna('')

            # 初始化特征列表
            all_features = []

            # 批处理处理文本
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size].tolist()

                # 使用tokenizer处理批次数据
                inputs = self.tokenizer(
                    batch_texts,
                    max_length=bert_config['max_length'],
                    padding=bert_config['padding'],
                    truncation=bert_config['truncation'],
                    return_tensors=bert_config['return_tensors']
                )

                # 将输入移动到正确的设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 使用新的autocast API
                use_amp = self.device.type == 'cuda'
                with torch.no_grad(), torch.amp.autocast(enabled=use_amp, device_type='cuda'):
                    outputs = self.bert_model(**inputs)

                # 获取[CLS]标记的输出作为特征
                batch_features = outputs.last_hidden_state[:, 0, :]
                all_features.append(batch_features.cpu())

                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 合并所有特征
            if all_features:
                features = torch.cat(all_features, dim=0)
                return features.to(self.device)
            return None

        except Exception as e:
            self.logger.error(f"BERT特征提取失败: {str(e)}")
            return None

    def _get_background_knowledge(self, stock: str, news_df: pd.DataFrame) -> str:
        """获取股票相关的背景知识"""
        try:
            template = f"""你是一个专业的金融分析师，请分析{stock}的以下特征：

1. 公司基本面：
{stock}公司的所属行业是什么
它的主营业务有哪些
它的市场地位怎么样

2. 相关方关系：
{stock}公司的主要竞争对手有哪些
它的上下游关系是什么样的
它所处的产业链位置

请基于新闻内容进行分析：
{chr(10).join(news_df['title'].tolist()) if not news_df.empty else '暂无相关新闻'}

请提供简洁的分析结果，每项特征用一句话概括。"""

            return self._get_openai_response(template)

        except Exception as e:
            self.logger.error(f"获取背景知识失败: {str(e)}")
            return ""

    def _extract_factors(self, stock: str, news_df: pd.DataFrame) -> List[str]:
        """提取影响股价的因素"""
        try:
            if news_df.empty:
                return []
            # 限制标题数量并合并
            max_titles = 10
            news_titles = news_df['title'].dropna().tolist()[:max_titles]
            news_text = '\n'.join(news_titles) if news_titles else '暂无相关新闻'
            template = f"""作为金融专家，请从以下新闻中提取可能影响{stock}股价的关键因素：

新闻内容：
{news_text}

请按以下类别分析影响因素：
1. 公司层面（如业绩、战略、管理层变动等）
2. 行业层面（如产业政策、供需关系、技术革新等）
3. 市场层面（如宏观经济、资金面、市场情绪等）

请列出最重要的{self.config['model']['k_factors']}个因素，每个因素需要：
1. 明确说明影响方向（利好/利空）
2. 给出影响程度（高/中/低）
3. 解释影响机制

请按"因素：影响方向（影响程度）- 影响机制"的格式输出。"""

            response = self._get_openai_response(template)
            factors = [f.strip() for f in response.split('\n') if f.strip()]
            return factors[:self.config['model']['k_factors']]  # 确保返回指定数量的因素

        except Exception as e:
            self.logger.error(f"提取因素失败: {str(e)}")
            return []

    def _prepare_price_history(self, price_df: pd.DataFrame) -> str:
        """准备价格历史文本
        Args:
            price_df: 价格数据
        Returns:
            价格历史文本
        """
        history = []
        for _, row in price_df.iterrows():
            date = row['date'].strftime('%Y-%m-%d')
            change = "上涨" if row['close'] > row['open'] else "下跌"
            history.append(f"在 {date}，股价{change}")
        return "\n".join(history)

    def _build_prediction_prompt(self, price_data: pd.DataFrame, news_data: pd.DataFrame,
                                 bert_features: torch.Tensor) -> str:
        try:
            # 1. 提取关键信息
            latest_date = price_data['date'].max().strftime('%Y-%m-%d')  # 格式化日期
            latest_price = price_data['close'].iloc[-1]
            price_change = (latest_price - price_data['close'].iloc[-2]) / price_data['close'].iloc[-2] * 100 if len(
                price_data) > 1 else 0

            # 2. 构建简洁的价格历史（最多5天）
            price_history = []
            for i in range(1, min(5, len(price_data))):
                date = price_data['date'].iloc[-i].strftime('%Y-%m-%d')
                price = price_data['close'].iloc[-i]
                previous_price = price_data['close'].iloc[-i - 1]
                change = (price - previous_price) / previous_price * 100
                price_history.append(f"{date}: {price:.2f} ({change:+.2f}%)")

            # 3. 提取关键新闻（最多3条，限制正文长度）
            recent_news = news_data.nlargest(3, 'publishedAt')
            news_summary = []
            for _, news in recent_news.iterrows():
                # 限制新闻正文长度
                content = news['title'][:100] if len(news['title']) > 100 else news['title']
                news_summary.append(f"- {content}")

            # 4. 构建简洁的提示
            prompt = f"""股票分析报告

    当前状态：
    - 日期: {latest_date}
    - 最新价格: {latest_price:.2f}
    - 涨跌幅: {price_change:+.2f}%

    近期走势：
    {chr(10).join(price_history)}

    重要新闻：
    {chr(10).join(news_summary)}

    请分析以上信息，预测未来走势。"""

            # 5. 检查并限制提示长度
            if len(prompt) > 2048:
                self.logger.warning(f"提示文本过长 ({len(prompt)} > 2048)，进行截断")
                # 保留关键信息
                prompt = prompt[:2048 - 3] + "..."

            return prompt

        except Exception as e:
            self.logger.error(f"构建提示失败: {str(e)}")
            return ""

    def _analyze_bert_features(self, features: torch.Tensor) -> str:
        """分析BERT特征"""
        try:
            # 获取BERT特征的均值，标准差，最大值等
            sentiment_score = features.mean().item()
            sentiment_std = features.std().item()
            sentiment_max = features.max().item()
            sentiment_min = features.min().item()

            # 更丰富的情感分析
            sentiment = '积极' if sentiment_score > 0 else '消极'
            sentiment_strength = '强' if abs(sentiment_score) > 0.5 else '弱'

            # 返回情感分析结果
            return f"新闻情感倾向：{sentiment} (得分: {sentiment_score:.2f}, 强度: {sentiment_strength}, 标准差: {sentiment_std:.2f}, 最大值: {sentiment_max:.2f}, 最小值: {sentiment_min:.2f})"

        except Exception as e:
            self.logger.error(f"BERT特征分析失败: {str(e)}")
            return "情感分析失败"

    def _get_prediction(self, prompt: str) -> Tuple[int, str]:
        """获取预测结果"""
        try:
            response = self._get_openai_response(prompt)

            # 解析预测结果
            prediction = 1 if "上涨" in response else 0

            return prediction, response

        except Exception as e:
            self.logger.error(f"获取预测结果失败: {str(e)}")
            return 0, str(e)

    def _calculate_confidence(self, factors: List[str], bert_features: torch.Tensor) -> float:
        """计算预测置信度"""
        # 结合因素数量和BERT特征计算置信度
        factor_confidence = min(len(factors) * 0.2, 0.5)
        bert_confidence = min(abs(bert_features.mean().item()), 0.5)
        return factor_confidence + bert_confidence

    def _get_default_prediction(self, date: datetime) -> Dict:
        """获取默认预测结果"""
        return {
            'date': date.strftime('%Y-%m-%d'),
            'prediction': 0,
            'confidence': 0.0,
            'reasoning': "预测失败",
            'knowledge': "",
            'factors': []
        }

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

            # 1. 获取背景知识
            knowledge = self._get_background_knowledge(symbol, news_df)

            # 2. 提取影响因素
            factors = self._extract_factors(symbol, news_df)

            # 3. 准备价格历史文本
            price_history = self._prepare_price_history(price_df)

            # 4. 构建最终预测提示
            prompt = self._build_prediction_prompt(
                price_data=price_df,
                news_data=news_df,
                bert_features=self._get_bert_features(news_df)
            )

            # 5. 获取预测结果
            prediction, reasoning = self._get_prediction(prompt)

            # 准备返回结果
            latest_date = price_df['date'].max()
            latest_date_str = latest_date.strftime('%Y-%m-%d') if latest_date else None

            return {
                'date': latest_date_str,
                'prediction': prediction,
                'confidence': self._calculate_confidence(factors, self._get_bert_features(news_df)),
                'reasoning': reasoning,
                'knowledge': knowledge,
                'factors': factors,
                'news_count': len(news_df)
            }

        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            return {
                'date': None,
                'prediction': 0,
                'confidence': 0.0,
                'reasoning': f"预测失败: {str(e)}",
                'knowledge': "",
                'factors': [],
                'news_count': 0
            }

    def evaluate_historical(self, market: str, symbol: str,
                            start_date: str, end_date: str) -> Dict:
        """评估历史预测性能"""
        try:
            self.logger.info(f"开始评估 {market} 市场 {symbol} 的历史表现...")

            # 1. 加载历史数据
            price_df, news_df = self.data_loader.load_data(
                market=market,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )

            if price_df.empty:
                raise ValueError(f"未找到 {symbol} 的价格数据")

            # 2. 初始化结果列表
            predictions = []
            actual = []
            dates = []
            confidences = []
            reasonings = []

            # 3. 逐日预测
            total_days = len(price_df) - 1

            # 确保日期索引是datetime类型
            price_df.index = pd.to_datetime(price_df.index)
            news_df.index = pd.to_datetime(news_df.index)

            for j in range(total_days):
                try:
                    current_date = price_df.index[j]
                    next_date = price_df.index[j + 1]

                    # 获取当前日期之前的数据
                    historical_price = price_df[:current_date]
                    historical_news = news_df[news_df.index <= current_date]

                    # 准备预测所需的数据
                    window_size = self.config['model']['window_size']
                    recent_price = historical_price[-window_size:] if len(
                        historical_price) > window_size else historical_price
                    recent_news = historical_news[-window_size:] if len(
                        historical_news) > window_size else historical_news

                    # 获取BERT特征
                    bert_features = self._get_bert_features(recent_news)

                    # 构建预测提示
                    prompt = self._build_prediction_prompt(
                        price_data=recent_price,
                        news_data=recent_news,
                        bert_features=bert_features
                    )

                    # 获取预测结果
                    prediction, reasoning = self._get_prediction(prompt)

                    # 计算实际涨跌
                    actual_movement = 1 if price_df.loc[next_date, 'close'] > price_df.loc[current_date, 'close'] else 0

                    # 记录结果
                    predictions.append(prediction)
                    actual.append(actual_movement)
                    dates.append(current_date.strftime('%Y-%m-%d'))
                    confidences.append(self._calculate_confidence(
                        self._extract_factors(symbol, recent_news),
                        bert_features
                    ))
                    reasonings.append(reasoning)

                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    self.logger.error(f"处理日期 {current_date} 时发生错误: {str(e)}")
                    continue

            # 4. 计算评估指标
            metrics = {
                'accuracy': accuracy_score(actual, predictions) if predictions else 0,
                'mcc': matthews_corrcoef(actual, predictions) if predictions else 0,
                'total_predictions': len(predictions)
            }

            # 5. 返回评估结果
            return {
                'predictions': predictions,
                'actual': actual,
                'dates': dates,
                'confidences': confidences,
                'metrics': metrics,
                'detailed_results': [
                    {
                        'date': d,
                        'prediction': p,
                        'actual': a,
                        'confidence': c,
                        'reasoning': r,
                        'correct': p == a
                    }
                    for d, p, a, c, r in zip(dates, predictions, actual, confidences, reasonings)
                ]
            }

        except Exception as e:
            self.logger.error(f"评估历史数据时发生错误: {str(e)}")
            return {
                'predictions': [],
                'actual': [],
                'dates': [],
                'confidences': [],
                'metrics': {
                    'accuracy': 0,
                    'mcc': 0,
                    'total_predictions': 0
                },
                'detailed_results': []
            }

    def __del__(self):
        """析构函数：清理GPU资源"""
        try:
            if hasattr(self, 'bert_model'):
                del self.bert_model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"清理GPU资源时出错: {str(e)}")

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