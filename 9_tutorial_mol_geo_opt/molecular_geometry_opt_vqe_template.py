# molecular_geometry_opt_vqe_template.py
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
import time
import itertools

# JAX配置
jax.config.update("jax_enable_x64", True)

def run_molecular_geometry_opt_vqe(config):
    """
    分子几何优化VQE算法参数化模板
    
    Args:
        config (dict): 包含symbols, initial_coordinates和可选配置参数的字典
            - symbols (list): 原子符号列表
            - initial_coordinates (list): 初始分子几何结构
            - charge (int, optional): 分子电荷，默认0
            - multiplicity (int, optional): 自旋多重度，默认1
            - max_iterations (int, optional): 最大迭代次数，默认50
            - circuit_lr (float, optional): 电路学习率，默认0.8
            - geometry_lr (float, optional): 几何学习率，默认0.8
            - gradient_threshold (float, optional): 梯度阈值，默认1e-5
            - energy_threshold (float, optional): 能量阈值，默认1e-8
            - finite_diff_step (float, optional): 有限差分步长，默认0.01
            - adaptive_circuit (bool, optional): 自适应电路，默认True
            - track_bond_lengths (bool, optional): 跟踪键长，默认True
            - max_excitations (int, optional): 最大激发数，默认10
            - print_progress (bool, optional): 打印进度，默认True
            - save_trajectory (bool, optional): 保存轨迹，默认True
    
    Returns:
        dict: 包含计算结果和完整配置信息的字典
    """
    
    # 1. 提取和验证输入参数
    symbols = config.get('symbols')
    initial_coordinates = config.get('initial_coordinates')
    
    if not symbols or not initial_coordinates:
        return {
            "success": False,
            "error": "缺少必需参数: symbols 和 initial_coordinates",
            "suggested_action": "请提供原子符号列表和初始几何结构"
        }
    
    # 2. 设置默认参数
    user_config = config.get('config', {})
    charge = user_config.get('charge', 0)
    multiplicity = user_config.get('multiplicity', 1)
    max_iterations = user_config.get('max_iterations', 50)
    circuit_lr = user_config.get('circuit_lr', 0.8)
    geometry_lr = user_config.get('geometry_lr', 0.8)
    gradient_threshold = user_config.get('gradient_threshold', 1e-5)
    energy_threshold = user_config.get('energy_threshold', 1e-8)
    finite_diff_step = user_config.get('finite_diff_step', 0.01)
    adaptive_circuit = user_config.get('adaptive_circuit', True)
    track_bond_lengths = user_config.get('track_bond_lengths', True)
    max_excitations = user_config.get('max_excitations', 10)
    print_progress = user_config.get('print_progress', True)
    save_trajectory = user_config.get('save_trajectory', True)
    
    # 3. 构建完整配置记录
    full_config = {
        "algorithm_name": "Molecular Geometry Optimization VQE",
        "algorithm_type": "molecular_geometry_optimization_vqe",
        "parameters_used": {
            "symbols": symbols,
            "initial_coordinates": initial_coordinates,
            "charge": charge,
            "multiplicity": multiplicity,
            "max_iterations": max_iterations,
            "circuit_lr": circuit_lr,
            "geometry_lr": geometry_lr,
            "gradient_threshold": gradient_threshold,
            "energy_threshold": energy_threshold,
            "finite_diff_step": finite_diff_step,
            "adaptive_circuit": adaptive_circuit,
            "track_bond_lengths": track_bond_lengths,
            "max_excitations": max_excitations,
            "print_progress": print_progress,
            "save_trajectory": save_trajectory
        },
        "default_values_applied": {
            "charge": charge == 0,
            "multiplicity": multiplicity == 1,
            "max_iterations": max_iterations == 50,
            "circuit_lr": circuit_lr == 0.8,
            "geometry_lr": geometry_lr == 0.8,
            "gradient_threshold": gradient_threshold == 1e-5,
            "energy_threshold": energy_threshold == 1e-8,
            "finite_diff_step": finite_diff_step == 0.01,
            "adaptive_circuit": adaptive_circuit == True,
            "track_bond_lengths": track_bond_lengths == True,
            "max_excitations": max_excitations == 10,
            "print_progress": print_progress == True,
            "save_trajectory": save_trajectory == True
        },
        "execution_environment": {
            "device": "default.qubit",
            "interface": "jax",
            "optimization_method": "joint_gradient_descent"
        }
    }
    
    try:
        # 4. 验证分子大小
        n_atoms = len(symbols)
        if n_atoms > 6:  # 经验限制
            return {
                "success": False,
                "error": f"分子过大({n_atoms}个原子)，几何优化VQE适用于小分子",
                "algorithm_config": full_config,
                "suggested_action": "请使用更小的分子"
            }
        
        # 5. 初始化几何结构
        coordinates = jnp.array(initial_coordinates)
        if coordinates.shape != (n_atoms, 3):
            return {
                "success": False,
                "error": f"坐标维度错误: 期望({n_atoms}, 3)，实际{coordinates.shape}",
                "algorithm_config": full_config,
                "suggested_action": "请提供正确格式的坐标矩阵"
            }
        
        # 6. 创建初始分子对象并验证
        try:
            molecule = qml.qchem.Molecule(symbols, coordinates, charge=charge)
            electrons = molecule.n_electrons
            orbitals = len(molecule.basis_set)
            qubits = 2 * orbitals  # 自旋轨道数
            
            if qubits > 8:
                return {
                    "success": False,
                    "error": f"需要{qubits}量子比特，超过几何优化VQE限制(8个)",
                    "algorithm_config": full_config,
                    "suggested_action": "请使用更小的分子或更小的基组"
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"创建分子对象失败: {str(e)}",
                "algorithm_config": full_config,
                "suggested_action": "请检查原子符号、坐标和电荷设置"
            }
        
        # 7. 定义参数化Hamiltonian构建函数
        def build_hamiltonian(coords):
            """构建参数化的分子Hamiltonian"""
            mol = qml.qchem.Molecule(symbols, coords, charge=charge)
            return qml.qchem.molecular_hamiltonian(mol)[0]
        
        # 8. 获取Hartree-Fock参考态
        hf_state = qml.qchem.hf_state(electrons, qubits)
        
        # 9. 自适应电路构建
        if adaptive_circuit:
            # 生成所有可能的激发
            all_singles, all_doubles = qml.qchem.excitations(electrons, qubits)
            
            # 限制激发数量
            selected_doubles = all_doubles[:min(len(all_doubles), max_excitations//2)]
            selected_singles = all_singles[:min(len(all_singles), max_excitations - len(selected_doubles))]
            
            if print_progress:
                print(f"自适应电路选择: {len(selected_doubles)}个双激发, {len(selected_singles)}个单激发")
        else:
            # 使用预定义的激发（基于分子类型）
            if len(symbols) == 3 and symbols.count('H') == 3:  # H3+
                selected_doubles = [[0, 1, 2, 3], [0, 1, 4, 5]]
                selected_singles = []
            else:
                # 通用情况
                all_singles, all_doubles = qml.qchem.excitations(electrons, qubits)
                selected_doubles = all_doubles[:2]
                selected_singles = all_singles[:2]
        
        n_params = len(selected_doubles) + len(selected_singles)
        
        # 10. 创建量子设备
        dev = qml.device("default.qubit", wires=qubits)
        
        # 11. 定义量子电路
        @qml.qnode(dev, interface="jax")
        def circuit(params, hamiltonian):
            # 初始化为Hartree-Fock态
            qml.BasisState(hf_state, wires=range(qubits))
            
            # 应用双激发门
            for i, wires in enumerate(selected_doubles):
                if i < len(params):
                    qml.DoubleExcitation(params[i], wires=wires)
            
            # 应用单激发门
            for i, wires in enumerate(selected_singles):
                param_idx = len(selected_doubles) + i
                if param_idx < len(params):
                    qml.SingleExcitation(params[param_idx], wires=wires)
            
            return qml.expval(hamiltonian)
        
        # 12. 定义成本函数
        def cost_function(params, coords):
            hamiltonian = build_hamiltonian(coords)
            return circuit(params, hamiltonian)
        
        # 13. 定义有限差分梯度计算
        def finite_diff_gradient(f, x, delta=None):
            """计算函数f关于x的有限差分梯度"""
            if delta is None:
                delta = finite_diff_step
            
            gradient = []
            x_flat = jnp.ravel(x)
            
            for i in range(len(x_flat)):
                shift = jnp.zeros_like(x_flat)
                shift = shift.at[i].set(0.5 * delta)
                
                x_plus = x_flat + shift
                x_minus = x_flat - shift
                
                grad_component = (f(x_plus.reshape(x.shape)) - f(x_minus.reshape(x.shape))) / delta
                gradient.append(grad_component)
            
            return jnp.array(gradient).reshape(x.shape)
        
        def geometry_gradient(params, coords):
            """计算几何坐标的梯度"""
            def energy_at_coords(coords):
                return cost_function(params, coords)
            
            return finite_diff_gradient(energy_at_coords, coords)
        
        # 14. 初始化优化参数
        circuit_params = jnp.zeros(n_params)
        current_coords = coordinates.copy()
        
        # 15. 初始化轨迹跟踪变量
        energy_history = []
        coords_history = []
        params_history = []
        gradient_history = []
        bond_length_history = []
        
        # 16. 执行优化
        start_time = time.time()
        converged = False
        convergence_reason = ""
        
        if print_progress:
            print("开始分子几何优化VQE算法...")
            print(f"分子: {' '.join(symbols)}, 电荷: {charge}")
            print(f"量子比特数: {qubits}, 电路参数数: {n_params}")
        
        for iteration in range(max_iterations):
            # 计算当前能量
            current_energy = cost_function(circuit_params, current_coords)
            energy_history.append(float(current_energy))
            
            if save_trajectory:
                coords_history.append(current_coords.copy())
                params_history.append(circuit_params.copy())
            
            # 计算电路参数梯度
            circuit_grad = jax.grad(cost_function, argnums=0)(circuit_params, current_coords)
            
            # 计算几何坐标梯度（力）
            coord_grad = geometry_gradient(circuit_params, current_coords)
            gradient_history.append(coord_grad.copy())
            
            # 计算键长（如果需要）
            if track_bond_lengths and n_atoms >= 2:
                bond_length = jnp.linalg.norm(current_coords[0] - current_coords[1]) * 0.529177210903  # 转换为埃
                bond_length_history.append(float(bond_length))
            
            # 检查梯度收敛
            max_gradient = jnp.max(jnp.abs(coord_grad))
            
            if print_progress and iteration % 4 == 0:
                if track_bond_lengths and bond_length_history:
                    print(f"步骤 {iteration}: E = {current_energy:.8f} Ha, "
                          f"最大梯度 = {max_gradient:.2e}, 键长 = {bond_length_history[-1]:.5f} Å")
                else:
                    print(f"步骤 {iteration}: E = {current_energy:.8f} Ha, "
                          f"最大梯度 = {max_gradient:.2e}")
            
            # 检查收敛条件
            if max_gradient <= gradient_threshold:
                converged = True
                convergence_reason = f"梯度收敛: 最大梯度 {max_gradient:.2e} <= {gradient_threshold}"
                break
            
            if iteration > 0:
                energy_change = abs(energy_history[-1] - energy_history[-2])
                if energy_change <= energy_threshold:
                    converged = True
                    convergence_reason = f"能量收敛: 能量变化 {energy_change:.2e} <= {energy_threshold}"
                    break
            
            # 更新参数
            circuit_params = circuit_params - circuit_lr * circuit_grad
            current_coords = current_coords - geometry_lr * coord_grad
        
        end_time = time.time()
        
        if print_progress:
            if converged:
                print(f"\n优化收敛! {convergence_reason}")
            else:
                print(f"\n达到最大迭代次数 ({max_iterations})，未完全收敛")
        
        # 17. 键长分析
        bond_analysis = {}
        if track_bond_lengths:
            if n_atoms >= 2:
                # 计算所有原子对的距离
                distances = {}
                for i in range(n_atoms):
                    for j in range(i+1, n_atoms):
                        dist = jnp.linalg.norm(current_coords[i] - current_coords[j]) * 0.529177210903
                        distances[f"{symbols[i]}{i+1}-{symbols[j]}{j+1}"] = float(dist)
                
                bond_analysis = {
                    "final_distances": distances,
                    "bond_length_history": bond_length_history,
                    "primary_bond_length": bond_length_history[-1] if bond_length_history else None
                }
        
        # 18. 构建返回结果
        result = {
            "success": True,
            "molecule": {
                "symbols": symbols,
                "initial_coordinates": initial_coordinates,
                "n_atoms": n_atoms,
                "charge": charge,
                "multiplicity": multiplicity,
                "n_electrons": electrons
            },
            "optimized_energy": float(energy_history[-1]),
            "energy_unit": "Hartree",
            "optimized_geometry": current_coords.tolist(),
            "optimized_circuit_params": circuit_params.tolist(),
            "final_gradients": coord_grad.tolist(),
            
            "convergence_info": {
                "converged": converged,
                "convergence_reason": convergence_reason,
                "iterations_performed": len(energy_history),
                "max_iterations_allowed": max_iterations,
                "final_max_gradient": float(max_gradient),
                "gradient_threshold": gradient_threshold,
                "energy_improvement": energy_history[0] - energy_history[-1] if len(energy_history) > 1 else 0.0
            },
            
            "optimization_trajectory": {
                "energy_history": energy_history,
                "coordinates_history": [c.tolist() for c in coords_history] if save_trajectory else "not_saved",
                "circuit_params_history": [p.tolist() for p in params_history] if save_trajectory else "not_saved",
                "gradient_history": [g.tolist() for g in gradient_history],
                "trajectory_saved": save_trajectory
            },
            
            "bond_analysis": bond_analysis,
            
            "circuit_info": {
                "qubits_used": qubits,
                "n_circuit_parameters": n_params,
                "double_excitations": selected_doubles,
                "single_excitations": selected_singles,
                "adaptive_circuit_used": adaptive_circuit,
                "hartree_fock_reference": hf_state.tolist()
            },
            
            "algorithm_config": full_config,
            
            "computational_details": {
                "optimization_method": "joint_gradient_descent",
                "gradient_calculation": {
                    "circuit_gradients": "analytical",
                    "geometry_gradients": "finite_difference",
                    "finite_diff_step": finite_diff_step
                },
                "learning_rates": {
                    "circuit": circuit_lr,
                    "geometry": geometry_lr
                },
                "execution_time": end_time - start_time,
                "avg_time_per_iteration": (end_time - start_time) / len(energy_history) if energy_history else 0
            }
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"分子几何优化VQE计算过程中发生错误: {str(e)}",
            "algorithm_config": full_config,
            "suggested_action": "请检查分子参数设置或JAX环境"
        }


def validate_molecular_geometry_opt_vqe_config(config):
    """
    验证分子几何优化VQE配置参数的辅助函数
    
    Args:
        config (dict): 配置字典
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(config, dict):
        return False, "配置必须是字典格式"
    
    if 'symbols' not in config or 'initial_coordinates' not in config:
        return False, "缺少必需参数: symbols 和 initial_coordinates"
    
    symbols = config['symbols']
    coordinates = config['initial_coordinates']
    
    if not isinstance(symbols, list) or not symbols:
        return False, "symbols必须是非空列表"
    
    if not isinstance(coordinates, list) or len(coordinates) != len(symbols):
        return False, "coordinates长度必须等于原子数量"
    
    # 检查坐标格式
    for i, coord in enumerate(coordinates):
        if not isinstance(coord, list) or len(coord) != 3:
            return False, f"原子{i+1}的坐标必须是长度为3的列表[x, y, z]"
        
        if not all(isinstance(x, (int, float)) for x in coord):
            return False, f"原子{i+1}的坐标必须都是数值"
    
    user_config = config.get('config', {})
    
    # 验证学习率
    for lr_name in ['circuit_lr', 'geometry_lr']:
        if lr_name in user_config:
            lr = user_config[lr_name]
            if not isinstance(lr, (int, float)) or lr <= 0:
                return False, f"{lr_name}必须是正数"
    
    # 验证阈值参数
    for threshold_name in ['gradient_threshold', 'energy_threshold']:
        if threshold_name in user_config:
            threshold = user_config[threshold_name]
            if not isinstance(threshold, (int, float)) or threshold <= 0:
                return False, f"{threshold_name}必须是正数"
    
    return True, ""


def estimate_molecular_geometry_opt_vqe_resources(symbols, charge=0):
    """
    估算分子几何优化VQE算法所需资源
    
    Args:
        symbols (list): 原子符号列表
        charge (int): 分子电荷
        
    Returns:
        dict: 资源估算结果
    """
    try:
        n_atoms = len(symbols)
        
        # 估算电子数和量子比特数
        electron_count = sum([_get_atomic_number(symbol) for symbol in symbols]) - charge
        estimated_qubits = n_atoms * 2  # 每个原子至少一个基函数
        
        # 估算激发数量
        singles, doubles = qml.qchem.excitations(electron_count, estimated_qubits)
        n_excitations = len(singles) + len(doubles)
        
        # 估算优化参数数量
        n_circuit_params = min(n_excitations, 10)  # 限制激发数量
        n_geometry_params = n_atoms * 3
        total_params = n_circuit_params + n_geometry_params
        
        # 估算计算时间
        time_per_iteration = 0.2 * total_params * estimated_qubits
        estimated_total_time = time_per_iteration * 50  # 假设50次迭代
        
        # 键长计算
        n_bonds = n_atoms * (n_atoms - 1) // 2
        
        return {
            "n_atoms": n_atoms,
            "electrons": electron_count,
            "qubits_required": estimated_qubits,
            "feasible": estimated_qubits <= 8 and n_atoms <= 6,
            "parameter_counts": {
                "circuit": n_circuit_params,
                "geometry": n_geometry_params,
                "total": total_params
            },
            "excitation_estimates": {
                "singles": len(singles) if estimated_qubits <= 10 else "too_many",
                "doubles": len(doubles) if estimated_qubits <= 10 else "too_many",
                "total_possible": n_excitations if estimated_qubits <= 10 else "too_many"
            },
            "optimization_features": {
                "joint_optimization": True,
                "adaptive_circuit": True,
                "bond_tracking": True,
                "gradient_methods": ["analytical_circuit", "finite_diff_geometry"]
            },
            "bond_analysis": {
                "n_possible_bonds": n_bonds,
                "bond_tracking_available": True
            },
            "estimated_time": f"{estimated_total_time:.1f}秒",
            "memory_requirement": "中等",
            "convergence_criteria": "化学精度梯度阈值"
        }
    except Exception as e:
        return {
            "feasible": False,
            "error": f"无法估算资源: {str(e)}"
        }


def _get_atomic_number(symbol):
    """获取原子序数"""
    atomic_numbers = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18
    }
    return atomic_numbers.get(symbol, 1)