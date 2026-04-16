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

## 模块说明

### 核心模块

| 模块 | 文件 | 功能描述 |
|------|------|----------|
| **公共工具** | `utils.py` | 路径常量定义、统一日志配置、MMD计算、KS检验、排列检验等统计函数 |
| **基准收集器** | `baseline_collector.py` | 使用 YOLO11 提取图像特征，生成 PCA 降维后的基准数据库 |
| **漂移检测器** | `drift_detector.py` | 四层级漂移检测：全局 MMD 检验、类别级漂移、特征级 KS 检验、样本级异常检测 |
| **报告生成器** | `drift_report.py` | 生成结构化的 JSON 漂移报告，包含统计结果和决策建议 |

### 分级与采样模块

| 模块 | 文件 | 功能描述 |
|------|------|----------|
| **颜色分级器** | `color_grader.py` | 基于 HSV 色彩空间的花卉颜色特征提取与分级，支持季节性参数调整 |
| **在线采样器** | `online_sampler.py` | 动态在线采样器，模拟生产环境中的数据流入，支持质量触发和时间间隔采样 |
| **质量分级器** | `quality_grader.py` | 集成 YOLO11 和 HSV 双路融合的实时分级服务，提供 FastAPI REST API |

### 训练与可视化模块

| 模块 | 文件 | 功能描述 |
|------|------|----------|
| **自动训练器** | `auto_trainer.py` | 自适应增量训练器，检测漂移后自动触发模型更新，支持模型融合 |
| **可视化模块** | `drift_visualizer.py` | 生成漂移分析图表，包括类别漂移热力图、特征重要性图、异常分布图等 |
| **动态检测流水线** | `dynamic_detection_pipeline.py` | 端到端演示流水线，模拟数据流、检测漂移、自动触发训练、生成报告 |

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
cd Adaptive_Flower_Grading_Project
pip install -r requirements.txt
```

### 依赖清单

```
matplotlib==3.10.8
numpy<2.0.0
pandas==2.3.3
scikit_learn==1.8.0
scipy==1.16.3
torch==2.6.0
ultralytics==8.3.49
fastapi==0.115.0
uvicorn==0.32.0
python-multipart==0.0.17
pillow==11.0.0
opencv-python>=4.8.0
```

---

## 快速开始

### 1. 准备花卉数据集

按以下结构组织数据：

```
dataset/
├── train/              # 训练集
│   ├── 1/             # 等级1（鲜艳饱满）
│   ├── 2/             # 等级2（颜色鲜艳）
│   ├── 3/             # 等级3（颜色一般）
│   └── 4/             # 等级4（颜色暗淡）
└── val/               # 验证集
    ├── 1/
    ├── 2/
    ├── 3/
    └── 4/
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 使用命令行工具

```bash
# 1. 收集基准资产（首次使用）
python main.py collect

# 2. 执行漂移检测
python main.py detect

# 3. 运行动态检测演示
python main.py demo

# 4. 生成可视化报告
python main.py visualize

# 5. 启动 REST API 服务
python main.py serve --port 8080

# 6. 分级单张图片
python main.py grade --image path/to/flower.jpg

# 7. 批量分级目录
python main.py grade --dir dataset/val/1
```

---

## 使用方法

### Step 1: 在线采样

`online_sampler.py` 负责实时采集花卉样本。

```python
from src.online_sampler import OnlineSampler

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
from src.auto_trainer import AutoTrainer

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
from src.quality_grader import QualityGrader

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

## 动态检测演示

RoseGrade 提供完整的动态数据检测演示流水线，用于模拟生产环境中的数据漂移场景并验证系统的自适应能力。

### 运行演示

```bash
python main.py demo
```

### 演示流程

演示流水线使用 `dataset/train/` 作为历史基准数据，`dataset/val/` 按窗口模拟新数据到达：

```
┌─────────────────────────────────────────────────────────────────┐
│                 动态检测演示流水线                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   基准数据 (train)        验证数据分窗口 (val)                    │
│   ┌─────────────┐        ┌─────────┬─────────┬─────────┐       │
│   │  历史数据    │        │ 窗口1   │ 窗口2   │ 窗口3   │ ...   │
│   │  (无扰动)   │        │ 无扰动  │ 无扰动  │ HSV扰动 │       │
│   └──────┬──────┘        └────┬────┴────┬────┴────┬────┘       │
│          │                    │         │         │            │
│          ▼                    ▼         ▼         ▼            │
│   ┌────────────────────────────────────────────────────────┐   │
│   │  1. 特征提取 → 2. 漂移检测 → 3. 判断 → 4. 增量训练(可选)  │   │
│   └────────────────────────────────────────────────────────┘   │
│                              │                                 │
│                              ▼                                 │
│   ┌────────────────────────────────────────────────────────┐   │
│   │  输出: 漂移趋势图 / JSON报告 / 训练前后对比              │   │
│   └────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 设计方案

| 步骤 | 说明 |
|------|------|
| **数据模拟** | 验证集分为多个时间窗口（默认4个窗口，每窗口约40张图片） |
| **HSV 扰动** | 在指定窗口（默认第3个窗口）对图像施加亮度/饱和度偏移，模拟季节变化 |
| **漂移检测** | 每个窗口进行 MMD 漂移检测，监控数据分布变化 |
| **自动训练** | 当检测到漂移超过阈值时，自动触发增量训练 |
| **效果验证** | 后续窗口使用更新后的模型，验证漂移是否缓解 |

### 输出结果

演示完成后生成以下输出：

- **漂移趋势图**: 显示各窗口 MMD 分数变化趋势
- **JSON 报告**: 包含详细的检测统计和决策信息
- **训练前后对比**: 验证增量训练的效果

### 高级配置

```bash
# 自定义窗口数量和扰动位置
python main.py demo --windows 6 --perturbation-window 3

# 自定义窗口大小和漂移阈值
python main.py demo --window-size 50 --drift-threshold 0.03
```

---

## API 文档

RoseGrade 提供基于 FastAPI 的 REST API 服务，支持 HTTP 请求进行花卉分级和漂移检测。

### 启动服务

```bash
python main.py serve --host 0.0.0.0 --port 8080
```

服务启动后，可通过以下地址访问：
- API 文档 (Swagger UI): http://localhost:8080/docs
- 健康检查: http://localhost:8080/api/health

### API 端点

#### POST /api/grade_single
单张图片分级

**请求**: `multipart/form-data`
- `file`: 图片文件 (JPEG/PNG)

**响应示例**:
```json
{
  "grade": 1,
  "confidence": 0.92,
  "fusion_score": 0.85,
  "yolo_result": {
    "grade": 1,
    "confidence": 0.95,
    "top5": [1, 2, 3, 4]
  },
  "hsv_result": {
    "grade": 1,
    "confidence": 0.88,
    "features": {
      "dominant_hue_name": "深红",
      "saturation_score": 0.82,
      "brightness_score": 0.75
    }
  }
}
```

---

#### POST /api/grade_batch
批量图片分级

**请求**: `multipart/form-data`
- `files`: 多个图片文件

**响应示例**:
```json
{
  "total": 10,
  "success": 10,
  "failed": 0,
  "results": [...],
  "timestamp": "2026-04-16T10:30:00"
}
```

---

#### GET /api/drift_status
获取漂移检测状态

**响应示例**:
```json
{
  "status": "stable",
  "severity": "none",
  "mmd_score": 0.0077,
  "is_drift": false,
  "drifted_classes": [],
  "report_generated_at": "2026-04-16T10:30:00",
  "report_path": "drift_report.json",
  "interpretation": "当前数据分布与训练阶段保持一致，未发现显著特征漂移，模型运行状态稳定。"
}
```

---

#### GET /api/drift_report
获取完整漂移报告

**响应**: JSON 格式的完整漂移检测报告，包含全局统计、类别级漂移、特征级漂移和样本级异常信息。

---

#### POST /api/trigger_detection
手动触发漂移检测

**响应示例**:
```json
{
  "success": true,
  "is_drift": false,
  "mmd_score": 0.0077,
  "status": "DATA STABLE",
  "report_path": "drift_report.json"
}
```

---

#### GET /api/health
健康检查

**响应示例**:
```json
{
  "status": "ok",
  "service": "RoseGrade",
  "version": "1.0.0",
  "timestamp": "2026-04-16T10:30:00"
}
```

---

#### GET /api/model_info
获取模型信息

**响应示例**:
```json
{
  "model_type": "YOLO11-cls",
  "num_classes": 4,
  "class_names": ["1", "2", "3", "4"],
  "model_path": "runs/classify/train2/weights/best.pt",
  "fused_model_available": false
}
```

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
RoseGrade/
├── src/                              # 核心源代码
│   ├── __init__.py                   # 包初始化与延迟导入
│   ├── utils.py                      # 公共工具（路径常量、日志、统计函数）
│   ├── baseline_collector.py         # YOLO11 特征提取与基准资产生成
│   ├── drift_detector.py             # 四层级漂移检测器
│   ├── drift_report.py               # JSON 结构化报告生成
│   ├── color_grader.py               # HSV 颜色特征提取与分级
│   ├── online_sampler.py             # 动态在线采样器
│   ├── auto_trainer.py               # 自适应增量训练器
│   ├── quality_grader.py             # 实时分级服务 (FastAPI)
│   ├── drift_visualizer.py           # 漂移分析可视化
│   └── dynamic_detection_pipeline.py # 动态数据检测流水线
├── main.py                           # 统一命令行入口
├── baseline_assets/                  # 持久化资产
│   ├── baseline_db.pkl               # 基准数据库
│   ├── pca_scaler.pkl                # 空间映射器
│   ├── val_test_data.pkl             # 验证集数据
│   └── drift_result.pkl              # 检测结果
├── dataset/                          # 数据集目录
│   ├── train/                        # 训练集
│   └── val/                          # 验证集
├── models/                           # 增量训练模型
├── reports/                          # 可视化报告输出
├── sampling_data/                    # 采样数据
├── runs/                             # YOLO 训练输出
├── drift_report.json                 # 漂移报告示例
├── requirements.txt                  # 依赖清单
└── README.md                         # 项目文档
```

---

## 路线图

- [x] 基准特征自动收集
- [x] MMD 全局漂移检测
- [x] 排列检验统计显著性
- [x] 多层级漂移分析（类别/特征/样本）
- [x] JSON 结构化报告生成
- [x] PCA 可视化模块
- [x] 在线流式检测支持
- [x] 自动告警与通知集成
- [x] Web Dashboard
- [x] 支持更多模型架构（RT-DETR、YOLO-World）

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
