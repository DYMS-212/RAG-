# Differentiable Hartree-Fock VQE Algorithm

## 基本信息
- **算法名称**: Differentiable Hartree-Fock VQE (可微分Hartree-Fock VQE)
- **函数名称**: run_differentiable_hf_vqe
- **算法类别**: 分子几何和基组联合优化
- **适用场景**: 分子几何优化、基组参数优化、力计算等量子化学应用

## 算法描述
可微分Hartree-Fock VQE算法结合了自动微分技术和VQE方法，可以同时优化三类参数：量子电路参数、分子几何坐标和基组参数。通过Hartree-Fock自洽场计算获得分子轨道，构建完全可微分的分子Hamiltonian，实现多参数联合优化。

## 技术规格
- **最大量子比特数**: 8
- **支持分子**: 小分子系统（H2, LiH, BeH2等）
- **基组支持**: STO-3G及其参数优化版本
- **可优化参数**: 电路参数、核坐标、基组指数、收缩系数
- **微分方法**: JAX自动微分
- **轨道可视化**: 支持原子轨道和分子轨道可视化

## 参数说明

### 必需参数
- **symbols**: 原子符号列表
- **initial_geometry**: 初始分子几何结构（原子单位）

### 可选参数
- **optimize_geometry**: 是否优化几何结构（默认True）
- **optimize_basis**: 是否优化基组参数（默认False）
- **initial_coeff**: 初始基组收缩系数（默认使用标准STO-3G）
- **initial_alpha**: 初始基组指数（默认使用标准STO-3G）
- **circuit_params**: 初始电路参数（默认全零）
- **max_iterations**: 最大优化迭代次数（默认50）
- **geometry_lr**: 几何优化学习率（默认0.5）
- **circuit_lr**: 电路参数学习率（默认0.25）
- **basis_lr**: 基组参数学习率（默认0.25）
- **force_threshold**: 力收敛阈值（默认1e-6）
- **visualize_orbitals**: 是否生成轨道可视化（默认False）

## 约束条件
- 需要JAX环境支持
- 分子大小受量子比特数限制
- 基组优化需要充分的初值设置

## 算法优势
- 真正的多参数联合优化
- 全自动微分，梯度计算精确高效
- 可获得比标准基组更低的能量
- 支持分子力计算和几何优化
- 提供轨道可视化功能
- 为量子化学提供了新的优化策略