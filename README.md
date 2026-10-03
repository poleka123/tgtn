# TGTN: Tucker-Guided Temporal Graph Network

基于 MTGNN 框架的时空图预测模型，通过引入全局可学习的 Tucker 张量分解，捕获节点、时间通道和空间因子之间的低秩交互关系。

**输入**: `[batch, nodes, input_steps, features]`  
**输出**: `[batch, nodes, output_steps]`

---

## 模型结构
输入 → Shape变换 → StartConv → [MTGNN层 × L] ├── 膨胀Inception时间卷积 (可选Tucker时间因子引导) ├── 图卷积 (mix-hop propagation) └── 残差连接 + LayerNorm → Skip连接 → 输出卷积 → 预测结果

### 核心组件

| 组件 | 说明 |
|------|------|
| `GlobalTuckerRepresentation` | 全局低秩张量因子 $U_N, U_T, U_C, G$ |
| `dilated_inception` | 多尺度膨胀卷积，支持时间因子门控 |
| `mixprop` | 混合传播图卷积 |
| `graph_constructor` | 可学习稀疏邻接矩阵构建 |

---

## 使用方法

### 安装依赖

```bash
pip install torch torchdiffeq numpy pandas scikit-learn

# TGTN
python main.py --model tgtn --filename pems08 --epochs 300

# 其他模型
python main.py --model mtgnn --filename pems08 --epochs 300
python main.py --model gru --filename pems08 --epochs 300

model/           # 模型实现
├── tgtn/       # TGTN核心 (net.py, layer.py)
├── mtgnn/      # MTGNN基础网络
└── ...         # 其他对比模型
methods/         # 训练评估 (train.py, evaluate.py)
data/            # 数据加载 (loader.py, graph_builder.py)
main.py          # 主入口
benchmark.py      # 批量测试