# LLMFactor

LLMFactor是一个基于大语言模型的股票市场预测框架，它通过分析新闻文本和历史股价数据来预测股票走势。

## 功能特点

- 使用顺序知识引导提示（SKGP）策略进行预测
- 分析新闻文本中的公司关系
- 提取可能影响股价的关键因素
- 结合历史价格数据进行预测
- 提供详细的预测推理过程

## 系统要求

- Python 3.8+
- Ollama（本地运行）
- llama3.1:8b-instruct-q8_0 模型

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/LLMFactor.git
cd LLMFactor
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 安装Ollama：
访问 [Ollama官网](https://ollama.ai/) 下载并安装Ollama。

4. 下载模型：
```bash
ollama pull llama3.1:8b-instruct-q8_0
```

5. 配置环境变量：
创建`.env`文件并添加以下内容：
```
NEWSAPI_KEY=your_newsapi_key
```

## 使用方法

1. 基本使用：
```python
from llm_factor import LLMFactor

# 初始化模型
model = LLMFactor(
    window_size=5,
    k_factors=5
)

# 进行预测
prediction, reasoning = model.predict_stock_movement(
    stock_target='NVDA',
    date_target='2024-03-01',
    news_target='相关新闻文本',
    price_history=[100, 101, 102, 103, 104],
    date_history=['2024-02-26', '2024-02-27', '2024-02-28', '2024-02-29', '2024-03-01']
)
```

2. 运行示例：
```bash
python example.py
```

## 评估指标

- 准确率（Accuracy）：预测正确的样本比例
- 马修斯相关系数（MCC）：考虑了所有混淆矩阵元素的评估指标

## 注意事项

1. 确保Ollama服务在本地运行（默认端口11434）
2. 需要有效的NewsAPI密钥
3. 确保有稳定的网络连接以获取股票数据和新闻
4. 预测结果仅供参考，不构成投资建议

## 引用

如果您在研究中使用了LLMFactor，请引用以下论文：

```
@article{llmfactor2024,
  title={LLMFactor: A Framework for Stock Market Prediction and Interpretation},
  author={Your Name},
  journal={Journal of Finance},
  year={2024}
}
```

## 许可证

MIT License 