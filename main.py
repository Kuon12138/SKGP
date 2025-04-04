import logging
import os
from predictor import Predictor
from data_loader import DataLoader
import json
from datetime import datetime
import yaml
from config import Config
from sklearn.metrics import accuracy_score, matthews_corrcoef


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('skgp.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"加载配置文件失败: {str(e)}")
        raise


def save_prediction(result: dict, stock_code: str, result_dir: str):
    """保存预测结果到文件"""
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


def evaluate_dataset(config: Config, stock_symbol: str, start_date: str, end_date: str):
    """
    评估模型在特定数据集上的性能

    Args:
        config: 配置对象
        stock_symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 初始化组件
        data_loader = DataLoader(config.model_config)
        predictor = Predictor(config.model_config)

        # 设置市场类型
        market = "CN" if stock_symbol.endswith(('.SZ', '.SH')) else "US"
        data_loader.set_market(market)

        # 加载数据
        stock_data, news_data = data_loader.load_dataset(stock_symbol, start_date, end_date)

        # 评估模型
        results = predictor.evaluate(stock_data, news_data, stock_symbol)

        # 打印结果
        logger.info(f"\n评估结果:")
        logger.info(f"准确率: {results['accuracy']:.2%}")
        logger.info(f"MCC: {results['mcc']:.2f}")
        logger.info(f"总预测次数: {results['total_predictions']}")
        logger.info(f"成功预测次数: {results['successful_predictions']}")

        # 进行单次预测
        if len(stock_data) < 6 or len(news_data) == 0:
            logger.warning("警告：数据量不足，无法进行预测")
            return results

        # 使用最后一天的数据进行预测
        date_target = stock_data.index[-1].strftime('%Y-%m-%d')
        price_history = stock_data['Close'].iloc[-6:-1].tolist()
        date_history = stock_data.index[-6:-1].strftime('%Y-%m-%d').tolist()

        # 获取当天的新闻
        day_news = [n for n in news_data if n['publishedAt'].strftime('%Y-%m-%d') == date_target]
        news_text = "\n".join([n['title'] + "\n" + n['description'] for n in day_news])

        if not news_text:
            logger.warning("警告：目标日期没有相关新闻")
            return results

        # 进行预测
        prediction, reasoning = predictor.predict_stock_movement(
            stock_symbol,
            date_target,
            news_text,
            price_history,
            date_history
        )

        logger.info(f"\n预测结果:")
        logger.info(f"日期: {date_target}")
        logger.info(f"预测: {'上涨' if prediction == 1 else '下跌'}")
        logger.info(f"推理: {reasoning}")

        # 添加预测结果到评估结果中
        results['last_prediction'] = {
            'date': date_target,
            'prediction': prediction,
            'reasoning': reasoning
        }

        return results

    except Exception as e:
        logger.error(f"评估过程中出错: {str(e)}")
        raise


def main():
    """主程序入口"""
    try:
        # 设置日志
        setup_logging()
        logging.info("SKGP预测系统启动...")
        
        # 加载配置
        config = load_config("config.yaml")
        logging.info("配置文件加载成功")
        
        # 初始化预测器
        predictor = Predictor()
        logging.info("预测器初始化完成")
        
        # 获取要预测的股票列表
        stocks = config['data']['stocks']
        
        # 测试中国市场预测
        logging.info("开始中国市场预测...")
        for stock in stocks['CN']:
            try:
                result = predictor.predict("CN", stock)
                logging.info(f"{stock} 预测完成")
                print(f"\n{stock} 预测结果:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                # 保存预测结果
                save_prediction(result, stock, config['data']['result_dir'])
            except Exception as e:
                logging.error(f"预测 {stock} 失败: {str(e)}")
                continue
        
        # 测试美国市场预测
        logging.info("开始美国市场预测...")
        for stock in stocks['US']:
            try:
                result = predictor.predict("US", stock)
                logging.info(f"{stock} 预测完成")
                print(f"\n{stock} 预测结果:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                # 保存预测结果
                save_prediction(result, stock, config['data']['result_dir'])
            except Exception as e:
                logging.error(f"预测 {stock} 失败: {str(e)}")
                continue
        
        logging.info("SKGP预测系统运行完成")
        
    except Exception as e:
        logging.error(f"程序运行失败: {str(e)}")
        raise


if __name__ == "__main__":
    main() 