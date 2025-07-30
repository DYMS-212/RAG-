# ADAPT-VQE Algorithm

## 基本信息
- **算法名称**: ADAPT-VQE (自适应变分量子本征求解器)
- **函数名称**: run_adapt_vqe_molecular
- **算法类别**: 分子基态能量计算
- **适用场景**: 中等规模分子的高效量子化学计算，支持稀疏Hamiltonian优化

## 算法描述
ADAPT-VQE使用PennyLane的AdaptiveOptimizer逐步构建量子电路，每次选择梯度最大的激发门并优化参数，直到收敛。支持稀疏Hamiltonian加速和手动分组选择策略。

## 技术规格
- **最大量子比特数**: 16
- **支持分子**: 中等规模分子（如LiH, BeH2, H2O等）
- **电路构建**: 逐门自适应添加或分组批量选择
- **优化策略**: 基于梯度大小的门选择
- **稀疏优化**: 支持SparseHamiltonian加速计算

## 参数说明

### 必需参数
- **symbols**: 原子符号列表
- **geometry**: 原子几何坐标（Bohr单位）

### 可选参数
- **active_electrons**: 活跃电子数（默认自动计算）
- **active_orbitals**: 活跃轨道数（默认自动计算）
- **method**: 构建方法（'adaptive_optimizer'或'manual'，默认'adaptive_optimizer'）
- **gradient_threshold**: 梯度选择阈值（默认1e-5，仅manual模式）
- **max_iterations**: 最大优化迭代次数（默认10）
- **learning_rate**: 优化器学习率（默认0.5）
- **optimizer**: 优化器类型（默认sgd）
- **use_sparse**: 是否使用稀疏Hamiltonian（默认False）
- **drain_pool**: 是否移除已选择的门（默认False）

## 约束条件
- 量子比特数不超过16个
- 稀疏模式仅支持parameter-shift微分方法
- manual模式需要指定gradient_threshold

## 算法优势
- 自动或手动控制电路构建过程
- 稀疏Hamiltonian显著加速大分子计算
- 灵活的门选择策略
- 高效的内存使用