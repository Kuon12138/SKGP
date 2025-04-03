from dataclasses import dataclass
from typing import Dict, Any
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """模型配置类"""
    window_size: int = 5
    k_factors: int = 5
    dataset: str = 'CMIN-US'
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.1:8b-instruct-q8_0"
    bert_model: str = 'bert-base-uncased'
    data_root: str = "Data"


@dataclass
class LogConfig:
    """日志配置类"""
    level: str = "INFO"
    format: str = '%(asctime)s - %(levelname)s - %(message)s'
    file: str = "logs/llm_factor.log"


@dataclass
class DataConfig:
    """数据配置类"""
    start_date: str = "2024-01-01"
    end_date: str = "2024-03-01"
    us_stock: str = "NVDA"
    cn_stock: str = "600519"
    market: str = ""


class Config:
    """总配置类"""
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.model_config = ModelConfig()
        self.log_config = LogConfig()
        self.data_config = DataConfig()
        self._load_config()

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """从YAML文件创建配置对象"""
        config = cls(config_path)
        return config

    def _load_config(self):
        """从YAML文件加载配置"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
                
                # 更新模型配置
                if 'model' in config_dict:
                    for key, value in config_dict['model'].items():
                        setattr(self.model_config, key, value)
                
                # 更新日志配置
                if 'logging' in config_dict:
                    for key, value in config_dict['logging'].items():
                        setattr(self.log_config, key, value)
                
                # 更新数据配置
                if 'data' in config_dict:
                    for key, value in config_dict['data'].items():
                        setattr(self.data_config, key, value)

    def save_config(self):
        """保存配置到YAML文件"""
        config_dict = {
            'model': self.model_config.__dict__,
            'logging': self.log_config.__dict__,
            'data': self.data_config.__dict__
        }
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False) 