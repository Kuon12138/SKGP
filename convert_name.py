from data_processing.name_converter import StockNameConverter
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 创建转换器
    converter = StockNameConverter()
    
    # 验证数据一致性
    converter.verify_data_consistency()
    
    # 确认是否继续
    response = input("是否继续转换文件名？(y/n): ")
    if response.lower() == 'y':
        # 执行转换
        converter.convert_news_filenames()
    else:
        logger.info("操作已取消")

if __name__ == "__main__":
    main()