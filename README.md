# 光伏电站功率预测小工具

本项目是一个基于机器学习的光伏发电功率预测实验案例。通过对 8 个不同装机容量的光伏站点数据进行集成训练，实现了一个能够根据实时工况预测输出功率的简易 Web 应用。

## 🌟 项目亮点
* **大数据量级**：处理并清洗了超过 54.3 万条真实的电站历史记录。
* **多站点通用**：一个模型兼容 30MW 到 130MW 不同容量等级的 8 个电站，具有良好的泛化性。
* **前后端闭环**：实现了从原始数据清洗、模型离线训练到交互式网页部署的全流程。

## 📊 核心技术指标
* **模型算法**：LightGBM 回归模型。
* **预测精度**：在 8 站点混合测试集上达到了 **$R^2 = 0.8251$**。
* **核心特征**：模型基于以下 5 个核心维度进行决策：
    1. 核心辐射度 (Irradiance)
    2. 电站编号 (Station ID)
    3. 装机容量 (Capacity)
    4. 月份 (Month)
    5. 小时 (Hour)

## 📁 项目结构说明
```text
Solar_Forecasting_Project/
├── data/                 # 原始 Excel 站点数据 (不上传 GitHub)
├── results/              # 训练产出
│   ├── models/           # 存储训练好的 lgbm_solar_full_model.pkl
│   └── plots/            # 存储特征重要性、预测对比图等
├── src/                  # 源代码存放
├── model_trainer2.py     # 核心训练脚本：负责数据清洗、训练及绘图
├── solar_web_app.py      # 网页端程序：基于 Streamlit 的交互式预测界面
└── requirements.txt      # 环境依赖清单
```

## 🚀 快速启动

### 1. 环境准备
确保已安装 Python 环境并安装依赖：
```bash
pip install -r requirements.txt
```

### 2. 模型训练
运行训练脚本生成模型文件和可视化图表：
```bash
python model_trainer2.py
```

### 3. 开启预测网页
使用 Streamlit 启动 Web 服务：
```bash
streamlit run solar_web_app.py --server.port 8000
```
启动后，通过浏览器访问 `http://localhost:8000` 即可进行实时工况模拟预测。

## 开发者札记
本项目在开发过程中克服了 Linux 环境下二进制库冲突、多站点数据特征对齐等技术难点。虽然作为新手项目尚有提升空间，但已完整实现了从数据到应用的转化过程。

