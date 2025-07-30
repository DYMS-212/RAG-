# VQE Algorithm

## 基本信息
- **算法名称**: VQE (Variational Quantum Eigensolver)
- **函数名称**: run_vqe_molecular
- **算法类别**: 分子基态能量计算
- **适用场景**: NISQ设备上的小分子量子化学计算

## 算法描述
使用变分量子本征求解器计算分子基态能量。基于Ritz变分原理，通过参数化量子电路准备试探态，优化Hamiltonian期望值找到基态。

## 技术规格
- **最大量子比特数**: 12
- **支持分子**: 基于量子比特限制的任意分子
- **变分电路**: DoubleExcitation ansatz
- **基组**: Minimal basis set
- **映射方式**: Jordan-Wigner变换

## 参数说明

### 必需参数
- **molecule**: 分子标识符(如H2, LiH, H2O等)

### 可选参数  
- **electrons**: 电子数(默认自动计算)
- **max_iterations**: 最大优化迭代次数(默认100)
- **conv_tol**: 收敛容忍度(默认1e-6)
- **learning_rate**: 优化器学习率(默认0.4)
- **optimizer**: 优化器类型(默认sgd)

## 约束条件
- 量子比特数不超过12个
- 仅支持PennyLane数据集中的分子
- 收敛不保证，依赖于分子复杂度

## 错误处理
- 超过量子比特限制时拒绝执行
- 不支持的分子格式时返回错误
- 优化不收敛时返回警告信息