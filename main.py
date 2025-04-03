import logging
from pathlib import Path
from config import Config
from data_loader import DataLoader
from predictor import Predictor


def setup_logging(log_config):
    """设置日志"""
    # 创建日志目录
    log_path = Path(log_config.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_config.level),
        format=log_config.format,
        handlers=[
            logging.FileHandler(log_config.file),
            logging.StreamHandler()
        ]
    )


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
    """主函数"""
    # 加载配置
    config = Config.from_yaml("config.yaml")
    
    # 设置日志
    setup_logging(config.log_config)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("\n开始评估模型...")
        
        # 根据market选择执行CN或US市场测试
        if config.data_config.market == "CN":
            logger.info(f"\n评估 {config.data_config.cn_stock} 数据集:")
            cn_results = evaluate_dataset(
                config,
                config.data_config.cn_stock,
                config.data_config.start_date,
                config.data_config.end_date
            )
            logger.info(f"\n{config.data_config.cn_stock} 评估结果:")
            logger.info(f"准确率: {cn_results['accuracy']:.2%}")
            logger.info(f"MCC: {cn_results['mcc']:.2%}")
            logger.info(f"总预测次数: {cn_results['total_predictions']}")
            logger.info(f"成功预测次数: {cn_results['successful_predictions']}")
        elif config.data_config.market == "US":
            logger.info(f"\n评估 {config.data_config.us_stock} 数据集:")
            us_results = evaluate_dataset(
                config,
                config.data_config.us_stock,
                config.data_config.start_date,
                config.data_config.end_date
            )
            logger.info(f"\n{config.data_config.us_stock} 评估结果:")
            logger.info(f"准确率: {us_results['accuracy']:.2%}")
            logger.info(f"MCC: {us_results['mcc']:.2%}")
            logger.info(f"总预测次数: {us_results['total_predictions']}")
            logger.info(f"成功预测次数: {us_results['successful_predictions']}")
        else:
            logger.error(f"不支持的市场类型: {config.data_config.market}")
            return
            
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise


if __name__ == "__main__":
    main() 