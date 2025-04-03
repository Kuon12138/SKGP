import pandas as pd
from pathlib import Path
import logging
import shutil
from typing import Dict, List, Set
from .stock_mapping import STOCK_NAME_MAPPING

class StockNameConverter:
    """股票名称转换器"""
    def __init__(self, data_root: str = "Data"):
        self.data_root = Path(data_root)
        self.logger = logging.getLogger(__name__)
        self.name_to_code = STOCK_NAME_MAPPING
        
    def convert_news_filenames(self) -> None:
        """转换新闻文件名"""
        news_dir = self.data_root / "CMIN-CN" / "news" / "raw"
        
        # 确保目录存在
        if not news_dir.exists():
            self.logger.error(f"目录不存在: {news_dir}")
            return
        
        # 获取所有csv文件
        csv_files = list(news_dir.glob("*.csv"))
        
        # 创建备份
        self._backup_files(news_dir)
        
        # 转换文件名
        for file_path in csv_files:
            self._convert_single_file(file_path)
    
    def _backup_files(self, directory: Path) -> None:
        """备份文件"""
        backup_dir = directory.parent / "raw_backup"
        if not backup_dir.exists():
            shutil.copytree(directory, backup_dir)
            self.logger.info(f"已创建备份: {backup_dir}")
    
    def _convert_single_file(self, file_path: Path) -> None:
        """转换单个文件名"""
        original_name = file_path.stem
        
        if original_name in self.name_to_code:
            new_name = self.name_to_code[original_name] + '.csv'
            new_path = file_path.parent / new_name
            
            try:
                shutil.move(str(file_path), str(new_path))
                self.logger.info(f"已重命名: {original_name} -> {new_name}")
            except Exception as e:
                self.logger.error(f"重命名失败 {original_name}: {str(e)}")
        else:
            self.logger.warning(f"未找到对应的股票代码: {original_name}")
    
    def verify_data_consistency(self) -> None:
        """验证数据一致性"""
        price_dir = self.data_root / "CMIN-CN" / "price" / "raw"
        news_dir = self.data_root / "CMIN-CN" / "news" / "raw"
        
        price_codes = self._get_file_codes(price_dir)
        news_codes = self._get_file_codes(news_dir)
        
        self._report_consistency(price_codes, news_codes)
    
    def _get_file_codes(self, directory: Path) -> Set[str]:
        """获取目录中的文件代码"""
        return set(f.stem for f in directory.glob("*.csv"))
    
    def _report_consistency(self, price_codes: Set[str], news_codes: Set[str]) -> None:
        """报告数据一致性"""
        only_in_price = price_codes - news_codes
        only_in_news = news_codes - price_codes
        
        if only_in_price:
            self.logger.warning(f"仅在price目录中存在的股票: {only_in_price}")
        if only_in_news:
            self.logger.warning(f"仅在news目录中存在的股票: {only_in_news}")