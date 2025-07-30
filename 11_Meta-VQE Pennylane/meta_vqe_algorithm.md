# Meta-VQE Algorithm

## 基本信息
- **算法名称**: Meta-VQE (Meta-Variational Quantum Eigensolver)
- **函数名称**: run_meta_vqe
- **算法类别**: 参数化Hamiltonian学习和预测
- **适用场景**: 参数扫描、相变研究、Hamiltonian参数依赖性分析

## 算法描述
Meta-VQE是一种元学习变分量子算法，通过将参数化Hamiltonian H(λ)的参数λ编码到变分ansatz中，实现一次训练预测多个参数值的基态能量。算法包含编码层(将Hamiltonian参数嵌入)和处理层(纯变分参数)，能够学习参数化系统的能量轮廓。

## 技术规格
- **最大量子比特数**: 10
- **支持系统**: 参数化自旋链、分子系统、格点模型
- **编码方式**: 线性编码、非线性编码
- **训练策略**: 多点训练，单次预测
- **典型应用**: XXZ自旋链、横场Ising模型、分子势能面

## 参数说明

### 必需参数
- **hamiltonian_type**: Hamiltonian类型（'xxz_spin_chain', 'ising', 'molecular'等）
- **parameter_name**: 被编码的参数名称（如'delta', 'field_strength'等）
- **training_points**: 训练参数值列表

### 可选参数
- **n_qubits**: 量子比特数（默认4）
- **encoding_type**: 编码方式（默认'linear'）
- **n_encoding_layers**: 编码层数（默认1）
- **n_processing_layers**: 处理层数（默认3）
- **training_epochs**: 训练轮数（默认100）
- **learning_rate**: 学习率（默认0.01）
- **optimizer**: 优化器类型（默认'adam'）
- **test_points**: 测试参数值列表（默认20个随机点）
- **fixed_params**: 固定的其他参数（如eta等）

## 约束条件
- 需要定义参数化Hamiltonian的具体形式
- 训练点应覆盖感兴趣的参数范围
- 编码层设计需要适配参数的物理意义

## 算法优势
- 一次训练，多次预测，效率远高于多次VQE
- 能够学习参数化系统的全局行为
- 适用于相变点检测和临界现象研究
- 支持参数插值和外推
- 为量子系统参数依赖性研究提供新工具