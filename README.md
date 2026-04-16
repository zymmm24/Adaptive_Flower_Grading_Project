# RoseGrade

**自适应玫瑰花卉分级软件系统**

RoseGrade 是一个面向鲜切花卉的智能分级系统，专门解决玫瑰花等切花在季节变化时颜色分级精度误差偏离较大的问题。传统花卉分级系统依赖人工采样和离线训练，无法动态适应环境变化（如光照、季节、生长条件等因素）导致的花卉颜色变化。

RoseGrade 通过动态在线采样、自动模型训练和实时模型更新技术，实现对玫瑰花等各类鲜切鲜花进行精准高效地自动化智能化分级分类。

---

## 适用场景

- **花卉分级**：玫瑰、康乃馨、百合等切花的自动化颜色分级
- **鲜花电商**：在线鲜花品质评估和智能定价
- **花卉贸易**：进口花卉的品质分级和质量控制
- **农业生产**：花卉种植过程的质量监控和分级
- **花店零售**：实时花卉品质评估和客户推荐

---

## 核心功能

| 功能模块 | 描述 |
|----------|------|
| 动态在线采样 | 实时采集花卉图像样本，支持多种采样策略 |
| 自适应颜色分级 | 基于HSV颜色空间的多维度花卉颜色分级算法 |
| 自动模型训练 | 增量学习和在线学习，动态更新分级模型 |
| 季节变化适应 | 检测环境变化并自动调整分级标准 |
| 实时分级评估 | 支持单张和批量花卉图像的实时分级 |

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         RoseGrade Pipeline                               │
├───────────────────────────────────────────────────────────────────────── ┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                │
│  │  花卉图像     │    │   颜色分级   │    │   实时采样    │                │
│  │   样本库      │    │   模型库     |    │   数据流      │               │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌────────────────────────────────────────────────────────────┐          │
│  │                 Step 1: Online Sampler                     │          │
│  │  • 实时采样 → HSV特征提取 → 质量评估 → 样本筛选 → 伪标签决策   │          │
│  └────────────────────────────┬───────────────────────────────┘          │
│                               │                                          │
│                               ▼                                          │
│                    ┌─────────────────────┐                               │
│                    │   sampling_data/    │                               │
│                    │  ├─ grade_a.pkl     │                               │
│                    │  ├─ grade_b.pkl     │                               │
│                    │  ├─ grade_c.pkl     │                               │
│                    │  ├─ grade_d.pkl     |                               |
│                    |  └─ ...             │                               |
│                    └──────────┬──────────┘                               │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐          │
│  │              Step 2: Drift Detector                        │          │
│  │  • 持续监测数据分布是否整体偏移，给出“正常/警告/严重”结论，     |          |
|  |       若未发生显著漂移，则不更新模型                          |          |       
│  └────────────────────────────┬───────────────────────────────┘          │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐          │
│  │             Step 3: Adaptive Sampler                       │          │
│  │  • 确认漂移后，从候选数据里挑取“最有信息”的小批样本进行训练，   |          |
|  |    减少噪声与成本                                           |          |
│  └────────────────────────────┬───────────────────────────────┘          │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐          │
│  │              Step 4: Auto Trainer                          │          │
│  │     • 增量训练 → 模型更新 → 性能验证 → 模型融合               │          │
│  └────────────────────────────┬───────────────────────────────┘          │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐          │
│  │              Step 5: Quality Grader                        │          │
│  │     • 颜色分级 → 质量评估 → 趋势分析                         │          │
│  └────────────────────────────────────────────────────────────┘          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 安装

### 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐，支持 CPU 运行)

### 安装依赖

```bash
git clone https://github.com/zymmm24/driftkit.git
cd DriftKit
pip install -r requirements.txt
```

### 依赖清单

```
opencv-python==4.8.1.78
matplotlib==3.10.8
numpy==2.4.0
pandas==2.3.3
scikit_learn==1.8.0
scipy==1.16.3
torch==2.6.0
pillow==10.1.0
tqdm==4.66.1
```

---

## 快速开始

### 1. 准备花卉数据集

按以下结构组织数据：

```
dataset/
├── roses/              # 玫瑰花卉数据集
│   ├── grade_a/       # A级花卉（鲜艳饱满）
│   ├── grade_b/       # B级花卉（颜色鲜艳）
│   ├── grade_c/       # C级花卉（颜色一般）
│   └── grade_d/       # D级花卉（颜色暗淡）
├── carnations/        # 康乃馨数据集
└── lilies/           # 百合花数据集
```

### 2. 初始化分级模型

```bash
cd src

# Step 1: 初始化花卉分级器
python color_grader.py --init

# Step 2: 训练初始模型
python auto_trainer.py --train --dataset ../dataset/roses
```

### 3. 运行自适应分级系统

```bash
# Step 1: 启动在线采样
python online_sampler.py --start --camera 0

# Step 2: 启动自动训练（后台运行）
python auto_trainer.py --auto-update --interval 3600

# Step 3: 启动实时分级服务
python quality_grader.py --serve --port 8080
```

---

## 使用方法

### Step 1: 在线采样

`online_sampler.py` 负责实时采集花卉样本。

```python
from online_sampler import OnlineSampler

sampler = OnlineSampler(
    camera_id=0,
    sample_interval=30,  # 每30秒采样一次
    quality_threshold=0.8  # 质量阈值
)

# 启动采样
sampler.start_sampling()

# 获取最新样本
samples = sampler.get_recent_samples(hours=24)
```

**采样策略：**
- **时间间隔采样**: 定期采集样本
- **质量触发采样**: 检测到质量变化时触发
- **手动采样**: 人工指定采样时机

### Step 2: 自动训练

`auto_trainer.py` 执行增量学习和模型更新。

```python
from auto_trainer import AutoTrainer

trainer = AutoTrainer(
    model_path="models/rose_grader.pkl",
    update_interval=3600  # 每小时更新一次
)

# 增量训练
trainer.incremental_train(new_samples="sampling_data/new_samples.pkl")

# 性能验证
performance = trainer.validate_model(test_data="dataset/val")

# 模型融合
trainer.model_fusion(old_model="models/rose_grader_v1.pkl",
                    new_model="models/rose_grader_v2.pkl")
```

**训练策略：**
- **在线学习**: 实时更新模型参数
- **增量学习**: 保留历史知识的同时学习新样本
- **模型融合**: 结合多个模型的优势

### Step 3: 质量分级

`quality_grader.py` 执行实时花卉分级。

```python
from quality_grader import QualityGrader

grader = QualityGrader(model_path="models/rose_grader.pkl")

# 单张图片分级
result = grader.grade_single("path/to/rose.jpg")

# 批量分级
results = grader.grade_batch(["img1.jpg", "img2.jpg", "img3.jpg"])

# 实时分级服务
grader.start_service(port=8080)
```

**分级结果结构：**

```python
{
    "grade": "A",                    # 分级结果 A/B/C/D
    "confidence": 0.92,             # 置信度
    "color_features": {             # 颜色特征
        "hue": 0.15,                # 色相
        "saturation": 0.85,         # 饱和度
        "brightness": 0.78          # 亮度
    },
    "quality_score": 8.5,           # 综合质量评分
    "recommendations": [...]        # 分级建议
}
```

---

## API 文档

### OnlineSampler

在线采样器类。

```python
class OnlineSampler:
    def __init__(self, camera_id: int = 0, sample_interval: int = 30)
    def start_sampling(self) -> None
    def stop_sampling(self) -> None
    def get_recent_samples(self, hours: int = 24) -> pd.DataFrame
    def trigger_manual_sample(self) -> dict
```

| 方法 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `__init__` | `camera_id`, `sample_interval` | - | 初始化采样器，设置摄像头和采样间隔 |
| `start_sampling` | - | - | 启动自动采样进程 |
| `get_recent_samples` | `hours` | `DataFrame` | 获取指定时间范围内的样本 |
| `trigger_manual_sample` | - | `dict` | 手动触发一次采样 |

### AutoTrainer

自动训练器类。

```python
class AutoTrainer:
    def __init__(self, model_path: str, update_interval: int = 3600)
    def incremental_train(self, new_samples: str) -> dict
    def validate_model(self, test_data: str) -> dict
    def model_fusion(self, old_model: str, new_model: str) -> str
    def start_auto_update(self) -> None
```

| 方法 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `incremental_train` | `new_samples` | `dict` | 对新样本进行增量训练 |
| `validate_model` | `test_data` | `dict` | 验证模型性能 |
| `model_fusion` | `old_model`, `new_model` | `str` | 融合新旧模型 |
| `start_auto_update` | - | - | 启动自动更新服务 |

### QualityGrader

质量分级器类。

```python
class QualityGrader:
    def __init__(self, model_path: str)
    def grade_single(self, image_path: str) -> dict
    def grade_batch(self, image_paths: list) -> list
    def start_service(self, port: int = 8080) -> None
    def get_seasonal_adjustment(self) -> dict
```

| 方法 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `grade_single` | `image_path` | `dict` | 对单张图片进行分级 |
| `grade_batch` | `image_paths` | `list` | 批量分级多张图片 |
| `start_service` | `port` | - | 启动分级Web服务 |
| `get_seasonal_adjustment` | - | `dict` | 获取季节性调整参数 |

---

## 报告结构示例

```json
{
  "meta": {
    "generated_at": "2025-12-31T20:06:13",
    "report_type": "YOLO Feature Drift Report",
    "version": "v1.0"
  },
  "data_info": {
    "baseline_size": 640,
    "test_size": 162
  },
  "statistics": {
    "mmd_score": 0.0077,
    "alpha": 0.05
  },
  "decision": {
    "is_drift": false,
    "status": "DATA STABLE"
  },
  "interpretation": "当前数据分布与训练阶段保持一致，未发现显著特征漂移，模型运行状态稳定。",
  "per_class_drift": {
    "1": { "mmd": 0.029, "is_drift": true },
    "2": { "mmd": 0.023, "is_drift": true }
  },
  "feature_level_drift": {
    "changed_dims": ["dim_106"]
  },
  "sample_level_drift": [
    { "img_name": "xxx.jpg", "nn_dist": 14.18 }
  ]
}
```

---

## 技术原理

### MMD (Maximum Mean Discrepancy)

MMD 是一种非参数的两样本检验方法，通过比较两个分布在再生核希尔伯特空间 (RKHS) 中的均值嵌入来判断它们是否来自同一分布。

```
MMD²(P, Q) = E[k(X, X')] + E[k(Y, Y')] - 2E[k(X, Y)]
```

其中 `k(·, ·)` 为 RBF 核函数。

### 排列检验 (Permutation Test)

通过随机打乱样本标签（来自基准/测试）计算零假设下的 MMD 分布，从而得到观测 MMD 的 p-value。

### 多层级漂移分析

| 层级 | 方法 | 用途 |
|------|------|------|
| 全局 | MMD + 排列检验 | 整体分布是否变化 |
| 类别 | 按类 MMD | 哪些类别发生漂移 |
| 特征 | KS 检验 + Cohen's d | 哪些特征维度异常 |
| 样本 | 最近邻距离 | 识别离群样本 |

---

## 项目结构

```
DriftKit/
├── src/                          # 核心源代码
│   ├── __init__.py
│   ├── baseline_collector.py     # 基准特征收集器
│   ├── drift_detector.py         # 漂移检测器
│   └── drift_report.py           # 报告生成器
├── baseline_assets/              # 持久化资产
│   ├── baseline_db.pkl           # 基准数据库
│   ├── pca_scaler.pkl            # 空间映射器
│   ├── val_test_data.pkl         # 验证集数据
│   └── drift_result.pkl          # 检测结果
├── dataset/                      # 数据集目录
│   ├── train/                    # 训练集
│   └── val/                      # 验证集
├── runs/                         # YOLO 训练输出
├── drift_report.json             # 漂移报告示例
├── requirements.txt              # 依赖清单
└── README.md                     # 项目文档
```

---

## 路线图

- [x] 基准特征自动收集
- [x] MMD 全局漂移检测
- [x] 排列检验统计显著性
- [x] 多层级漂移分析（类别/特征/样本）
- [x] JSON 结构化报告生成
- [ ] PCA 可视化模块
- [ ] 在线流式检测支持
- [ ] 自动告警与通知集成
- [ ] Web Dashboard
- [ ] 支持更多模型架构（RT-DETR、YOLO-World）

---

## 贡献

欢迎贡献代码、报告问题或提出改进建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 许可证

本项目采用 MIT 许可证。
