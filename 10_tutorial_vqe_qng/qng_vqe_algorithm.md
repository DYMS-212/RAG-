# Quantum Natural Gradient VQE Algorithm

## 基本信息
- **算法名称**: Quantum Natural Gradient VQE (量子自然梯度VQE)
- **函数名称**: run_qng_vqe
- **算法类别**: 加速优化的变分量子本征求解器
- **适用场景**: 需要快速收敛的VQE计算、参数优化困难的变分电路

## 算法描述
量子自然梯度VQE使用Fubini-Study度量张量来重新调整参数空间中的梯度方向，实现比标准梯度下降更快的收敛。通过考虑量子态空间的几何结构，QNG能够选择更优的优化路径，特别适用于参数景观复杂的变分量子电路。

## 技术规格
- **最大量子比特数**: 12
- **支持分子**: 小到中等规模分子和量子系统
- **度量张量近似**: 对角近似、块对角近似
- **优化器**: QNGOptimizer with Fubini-Study metric
- **电路支持**: 任意参数化量子电路
- **收敛优势**: 通常比标准梯度下降快2-5倍收敛

## 参数说明

### 必需参数
- **hamiltonian**: 目标Hamiltonian（Pauli字符串或qml.Hamiltonian）
- **ansatz**: 变分量子电路ansatz

### 可选参数
- **step_size**: 学习步长（默认0.01）
- **metric_approx**: 度量张量近似方法（默认'block-diag'）
- **regularization**: 正则化参数λ（默认0.001）
- **max_iterations**: 最大优化迭代次数（默认500）
- **convergence_threshold**: 收敛阈值（默认1e-6）
- **initial_params**: 初始参数（默认随机初始化）
- **compare_standard**: 是否与标准梯度下降对比（默认True）
- **track_metric_condition**: 是否跟踪度量张量条件数（默认False）

## 约束条件
- 电路必须支持参数微分
- 度量张量计算需要额外的量子计算开销
- 参数数量影响度量张量计算复杂度

## 算法优势
- 收敛速度显著快于标准梯度下降
- 能够适应参数空间的量子几何结构
- 对初始参数选择更加鲁棒
- 特别适用于贫瘠高原(barren plateau)问题
- 提供更稳定的优化轨迹