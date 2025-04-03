import logging
from pathlib import Path
from config import Config
from data_loader import DataLoader
from predictor import Predictor
import os
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, matthews_corrcoef


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
    """主函数：执行SKGP算法的股票预测流程"""
    try:
        # 1. 初始化预测器和日志
        predictor = Predictor()
        logging.info("初始化SKGP预测系统...")
        
        # 2. 获取配置中的股票列表
        stocks = predictor.config['data'].get('stocks', {
            'CN': [predictor.config['data'].get('cn_stock', '600519.SH')],
            'US': [predictor.config['data'].get('us_stock', 'AAPL')]
        })
        
        # 3. 创建结果目录
        result_dir = "predictions"
        os.makedirs(result_dir, exist_ok=True)
        
        # 4. 初始化评估指标统计
        all_predictions = []
        all_actual = []
        market_metrics = {
            'CN': {'predictions': [], 'actual': []},
            'US': {'predictions': [], 'actual': []}
        }
        
        # 5. 处理中国市场
        logging.info("\n开始处理中国市场股票...")
        cn_results = {}
        for stock in stocks['CN']:
            try:
                logging.info(f"\n分析股票: {stock}")
                # 获取历史数据进行评估
                evaluation_result = predictor.evaluate_historical(
                    market="CN",
                    symbol=stock,
                    start_date=predictor.config['data']['start_date'],
                    end_date=predictor.config['data']['end_date']
                )
                
                # 收集评估数据
                if evaluation_result:
                    market_metrics['CN']['predictions'].extend(evaluation_result['predictions'])
                    market_metrics['CN']['actual'].extend(evaluation_result['actual'])
                    all_predictions.extend(evaluation_result['predictions'])
                    all_actual.extend(evaluation_result['actual'])
                
                # 进行当前预测
                logging.info("进行当前预测...")
                result = predictor.predict("CN", stock)
                
                # 记录分析阶段和保存结果
                cn_results[stock] = {
                    'current_prediction': result,
                    'historical_evaluation': evaluation_result
                }
                
                # 保存详细结果
                result_file = os.path.join(result_dir, f"CN_{stock}_{datetime.now().strftime('%Y%m%d')}.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(cn_results[stock], f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                logging.error(f"处理股票 {stock} 时发生错误: {str(e)}")
                continue
        
        # 6. 处理美国市场
        logging.info("\n开始处理美国市场股票...")
        us_results = {}
        for stock in stocks['US']:
            try:
                logging.info(f"\n分析股票: {stock}")
                # 获取历史数据进行评估
                evaluation_result = predictor.evaluate_historical(
                    market="US",
                    symbol=stock,
                    start_date=predictor.config['data']['start_date'],
                    end_date=predictor.config['data']['end_date']
                )
                
                # 收集评估数据
                if evaluation_result:
                    market_metrics['US']['predictions'].extend(evaluation_result['predictions'])
                    market_metrics['US']['actual'].extend(evaluation_result['actual'])
                    all_predictions.extend(evaluation_result['predictions'])
                    all_actual.extend(evaluation_result['actual'])
                
                # 进行当前预测
                logging.info("进行当前预测...")
                result = predictor.predict("US", stock)
                
                # 记录分析阶段和保存结果
                us_results[stock] = {
                    'current_prediction': result,
                    'historical_evaluation': evaluation_result
                }
                
                # 保存详细结果
                result_file = os.path.join(result_dir, f"US_{stock}_{datetime.now().strftime('%Y%m%d')}.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(us_results[stock], f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                logging.error(f"处理股票 {stock} 时发生错误: {str(e)}")
                continue
        
        # 7. 计算整体评估指标
        metrics = {
            'overall': {
                'accuracy': accuracy_score(all_actual, all_predictions) if all_predictions else 0,
                'mcc': matthews_corrcoef(all_actual, all_predictions) if all_predictions else 0,
                'total_predictions': len(all_predictions)
            },
            'CN': {
                'accuracy': accuracy_score(
                    market_metrics['CN']['actual'], 
                    market_metrics['CN']['predictions']
                ) if market_metrics['CN']['predictions'] else 0,
                'mcc': matthews_corrcoef(
                    market_metrics['CN']['actual'], 
                    market_metrics['CN']['predictions']
                ) if market_metrics['CN']['predictions'] else 0,
                'total_predictions': len(market_metrics['CN']['predictions'])
            },
            'US': {
                'accuracy': accuracy_score(
                    market_metrics['US']['actual'], 
                    market_metrics['US']['predictions']
                ) if market_metrics['US']['predictions'] else 0,
                'mcc': matthews_corrcoef(
                    market_metrics['US']['actual'], 
                    market_metrics['US']['predictions']
                ) if market_metrics['US']['predictions'] else 0,
                'total_predictions': len(market_metrics['US']['predictions'])
            }
        }
        
        # 8. 生成汇总报告
        logging.info("\n生成预测汇总报告...")
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics,
            'cn_market': {
                'total_stocks': len(stocks['CN']),
                'successful_predictions': len(cn_results),
                'results': cn_results
            },
            'us_market': {
                'total_stocks': len(stocks['US']),
                'successful_predictions': len(us_results),
                'results': us_results
            }
        }
        
        # 打印评估指标
        logging.info("\n模型评估指标:")
        logging.info("整体表现:")
        logging.info(f"准确率 (ACC): {metrics['overall']['accuracy']:.4f}")
        logging.info(f"马修斯相关系数 (MCC): {metrics['overall']['mcc']:.4f}")
        logging.info(f"总预测次数: {metrics['overall']['total_predictions']}")
        
        logging.info("\n中国市场:")
        logging.info(f"准确率 (ACC): {metrics['CN']['accuracy']:.4f}")
        logging.info(f"马修斯相关系数 (MCC): {metrics['CN']['mcc']:.4f}")
        logging.info(f"预测次数: {metrics['CN']['total_predictions']}")
        
        logging.info("\n美国市场:")
        logging.info(f"准确率 (ACC): {metrics['US']['accuracy']:.4f}")
        logging.info(f"马修斯相关系数 (MCC): {metrics['US']['mcc']:.4f}")
        logging.info(f"预测次数: {metrics['US']['total_predictions']}")
        
        # 保存汇总报告
        summary_file = os.path.join(result_dir, f"summary_{datetime.now().strftime('%Y%m%d')}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logging.info(f"\n汇总报告已保存至：{summary_file}")
        
        logging.info("\nSKGP预测流程完成！")
        
    except Exception as e:
        logging.error(f"执行SKGP预测流程时发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main() 