# Catalyst VQE Algorithm

## 基本信息
- **算法名称**: Catalyst VQE (JIT编译变分量子本征求解器)
- **函数名称**: run_catalyst_vqe_molecular
- **算法类别**: 分子基态能量计算
- **适用场景**: 需要高性能计算的量子化学VQE问题

## 算法描述
基于Catalyst框架的VQE算法，通过量子即时编译(QJIT)技术显著提升计算性能。使用JAX生态系统和Optax优化器，支持整个优化流程的JIT编译，在保持VQE算法准确性的同时大幅提升执行速度。

## 技术规格
- **最大量子比特数**: 12
- **支持分子**: 小到中等规模分子
- **编译框架**: Catalyst QJIT
- **后端**: JAX + Lightning.qubit
- **优化器**: Optax系列优化器
- **性能提升**: 相比标准VQE有显著性能提升

## 参数说明

### 必需参数
- **molecule**: 分子标识符或完整分子数据

### 可选参数
- **ansatz_type**: 变分电路类型(默认'double_excitation')
- **optimizer**: Optax优化器类型(默认'sgd')  
- **learning_rate**: 学习率(默认0.4)
- **max_iterations**: 最大迭代次数(默认10)
- **initial_params**: 初始参数(默认全零)
- **compile_optimization**: 是否编译整个优化循环(默认True)
- **diff_method**: 微分方法(默认'adjoint')

## 约束条件
- 需要Catalyst编译环境支持
- 必须使用JAX兼容的组件
- 量子比特数限制基于内存和编译时间

## 算法优势
- JIT编译显著提升执行速度
- 整个优化流程可编译优化
- 与JAX生态系统无缝集成
- 支持高级控制流编译
- 内存使用优化