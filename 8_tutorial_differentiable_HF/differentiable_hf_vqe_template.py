# differentiable_hf_vqe_template.py
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

# JAX配置
jax.config.update("jax_enable_x64", True)

def run_differentiable_hf_vqe(config):
    """
    可微分Hartree-Fock VQE算法参数化模板
    
    Args:
        config (dict): 包含symbols, initial_geometry和可选配置参数的字典
            - symbols (list): 原子符号列表
            - initial_geometry (list): 初始分子几何结构
            - optimize_geometry (bool, optional): 是否优化几何，默认True
            - optimize_basis (bool, optional): 是否优化基组，默认False
            - initial_coeff (list, optional): 初始基组收缩系数
            - initial_alpha (list, optional): 初始基组指数
            - circuit_params (list, optional): 初始电路参数
            - max_iterations (int, optional): 最大迭代次数，默认50
            - geometry_lr (float, optional): 几何学习率，默认0.5
            - circuit_lr (float, optional): 电路学习率，默认0.25
            - basis_lr (float, optional): 基组学习率，默认0.25
            - force_threshold (float, optional): 力阈值，默认1e-6
            - visualize_orbitals (bool, optional): 轨道可视化，默认False
            - print_progress (bool, optional): 打印进度，默认True
    
    Returns:
        dict: 包含计算结果和完整配置信息的字典
    """
    
    # 1. 提取和验证输入参数
    symbols = config.get('symbols')
    initial_geometry = config.get('initial_geometry')
    
    if not symbols or not initial_geometry:
        return {
            "success": False,
            "error": "缺少必需参数: symbols 和 initial_geometry",
            "suggested_action": "请提供原子符号列表和初始几何结构"
        }
    
    # 2. 设置默认参数
    user_config = config.get('config', {})
    optimize_geometry = user_config.get('optimize_geometry', True)
    optimize_basis = user_config.get('optimize_basis', False)
    initial_coeff = user_config.get('initial_coeff')
    initial_alpha = user_config.get('initial_alpha')
    circuit_params = user_config.get('circuit_params')
    max_iterations = user_config.get('max_iterations', 50)
    geometry_lr = user_config.get('geometry_lr', 0.5)
    circuit_lr = user_config.get('circuit_lr', 0.25)
    basis_lr = user_config.get('basis_lr', 0.25)
    force_threshold = user_config.get('force_threshold', 1e-6)
    visualize_orbitals = user_config.get('visualize_orbitals', False)
    print_progress = user_config.get('print_progress', True)
    
    # 3. 构建完整配置记录
    full_config = {
        "algorithm_name": "Differentiable Hartree-Fock VQE",
        "algorithm_type": "differentiable_hf_variational_quantum_eigensolver",
        "parameters_used": {
            "symbols": symbols,
            "initial_geometry": initial_geometry,
            "optimize_geometry": optimize_geometry,
            "optimize_basis": optimize_basis,
            "initial_coeff": initial_coeff,
            "initial_alpha": initial_alpha,
            "circuit_params": circuit_params,
            "max_iterations": max_iterations,
            "geometry_lr": geometry_lr,
            "circuit_lr": circuit_lr,
            "basis_lr": basis_lr,
            "force_threshold": force_threshold,
            "visualize_orbitals": visualize_orbitals,
            "print_progress": print_progress
        },
        "default_values_applied": {
            "optimize_geometry": optimize_geometry == True,
            "optimize_basis": optimize_basis == False,
            "max_iterations": max_iterations == 50,
            "geometry_lr": geometry_lr == 0.5,
            "circuit_lr": circuit_lr == 0.25,
            "basis_lr": basis_lr == 0.25,
            "force_threshold": force_threshold == 1e-6,
            "visualize_orbitals": visualize_orbitals == False,
            "print_progress": print_progress == True,
            "initial_coeff": initial_coeff is None,
            "initial_alpha": initial_alpha is None,
            "circuit_params": circuit_params is None
        },
        "execution_environment": {
            "device": "default.qubit",
            "interface": "jax",
            "automatic_differentiation": True
        }
    }
    
    try:
        # 4. 验证分子大小
        n_atoms = len(symbols)
        if n_atoms > 4:  # 经验限制
            return {
                "success": False,
                "error": f"分子过大({n_atoms}个原子)，可微分HF-VQE适用于小分子",
                "algorithm_config": full_config,
                "suggested_action": "请使用更小的分子或标准VQE算法"
            }
        
        # 5. 初始化几何结构
        geometry = jnp.array(initial_geometry)
        if geometry.shape[1] != 3:
            return {
                "success": False,
                "error": f"几何结构维度错误: 期望(N, 3)，实际{geometry.shape}",
                "algorithm_config": full_config,
                "suggested_action": "请提供正确格式的坐标：[[x1,y1,z1], [x2,y2,z2], ...]"
            }
        
        # 6. 创建初始分子对象
        try:
            mol = qml.qchem.Molecule(symbols, geometry)
            electrons = mol.n_electrons
            qubits = 2 * len(mol.basis_set)  # 自旋轨道数
            
            if qubits > 8:
                return {
                    "success": False,
                    "error": f"需要{qubits}量子比特，超过可微分HF-VQE限制(8个)",
                    "algorithm_config": full_config,
                    "suggested_action": "请使用更小的分子或更小的基组"
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"创建分子对象失败: {str(e)}",
                "algorithm_config": full_config,
                "suggested_action": "请检查原子符号和坐标格式"
            }
        
        # 7. 初始化基组参数
        if initial_coeff is None:
            coeff = mol.coeff  # 使用标准STO-3G系数
        else:
            coeff = jnp.array(initial_coeff)
        
        if initial_alpha is None:
            alpha = mol.alpha  # 使用标准STO-3G指数
        else:
            alpha = jnp.array(initial_alpha)
        
        # 8. 初始化电路参数
        if circuit_params is None:
            # 根据分子确定需要的激发门数量
            if electrons == 2 and qubits == 4:  # H2
                circuit_param = jnp.array([0.0])
            else:
                # 一般情况下估算
                n_excitations = min(electrons * (qubits - electrons), 5)
                circuit_param = jnp.zeros(n_excitations)
        else:
            circuit_param = jnp.array(circuit_params)
        
        # 9. 创建量子设备
        dev = qml.device("default.qubit", wires=qubits)
        
        # 10. 定义能量函数
        def energy_function():
            @qml.qnode(dev, interface="jax")
            def circuit(*args):
                # Hartree-Fock初态
                hf_state = qml.qchem.hf_state(electrons, qubits)
                qml.BasisState(hf_state, wires=range(qubits))
                
                # 激发门（根据分子类型调整）
                if len(args[0]) == 1:  # H2情况
                    qml.DoubleExcitation(args[0][0], wires=[0, 1, 2, 3])
                else:
                    # 更复杂分子的激发门
                    for i, param in enumerate(args[0]):
                        if i == 0 and qubits >= 4:
                            qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
                        elif i == 1 and qubits >= 6:
                            qml.DoubleExcitation(param, wires=[0, 1, 4, 5])
                
                # 构建分子和Hamiltonian
                mol = qml.qchem.Molecule(symbols, args[1], alpha=args[3], coeff=args[2])
                H = qml.qchem.molecular_hamiltonian(mol, args=args[1:])[0]
                return qml.expval(H)
            
            return circuit
        
        energy_circuit = energy_function()
        
        # 11. 执行优化
        start_time = time.time()
        
        energy_history = []
        geometry_history = []
        circuit_param_history = []
        force_history = []
        
        converged = False
        
        if print_progress:
            print("开始可微分Hartree-Fock VQE优化...")
            print(f"优化参数: 几何={optimize_geometry}, 基组={optimize_basis}")
        
        for iteration in range(max_iterations):
            # 构建参数列表
            args = [circuit_param, geometry, coeff, alpha]
            
            # 计算能量
            current_energy = energy_circuit(*args)
            energy_history.append(float(current_energy))
            geometry_history.append(geometry.copy())
            circuit_param_history.append(circuit_param.copy())
            
            # 计算所有梯度
            if optimize_geometry and optimize_basis:
                # 同时优化所有参数
                gradients = jax.grad(energy_circuit, argnums=[0, 1, 2, 3])(*args)
                
                # 更新参数
                circuit_param = circuit_param - circuit_lr * gradients[0]
                geometry = geometry - geometry_lr * gradients[1]
                coeff = coeff - basis_lr * gradients[2]
                alpha = alpha - basis_lr * gradients[3]
                
                forces = gradients[1]
                
            elif optimize_geometry:
                # 只优化几何和电路参数
                gradients = jax.grad(energy_circuit, argnums=[0, 1])(*args)
                
                circuit_param = circuit_param - circuit_lr * gradients[0]
                geometry = geometry - geometry_lr * gradients[1]
                
                forces = gradients[1]
                
            else:
                # 只优化电路参数
                gradient = jax.grad(energy_circuit, argnums=0)(*args)
                circuit_param = circuit_param - circuit_lr * gradient
                forces = jnp.zeros_like(geometry)
            
            force_history.append(forces.copy())
            max_force = jnp.abs(forces).max()
            
            # 打印进度
            if print_progress and iteration % 5 == 0:
                print(f"迭代 {iteration}: E = {current_energy:.8f} Ha, "
                      f"最大力 = {max_force:.8f}")
            
            # 检查收敛
            if optimize_geometry and max_force < force_threshold:
                converged = True
                if print_progress:
                    print(f"在第{iteration}步收敛！")
                break
            elif not optimize_geometry and iteration > 0:
                energy_change = abs(energy_history[-1] - energy_history[-2])
                if energy_change < 1e-8:
                    converged = True
                    if print_progress:
                        print(f"能量收敛在第{iteration}步！")
                    break
        
        end_time = time.time()
        
        # 12. 生成轨道可视化数据（如果需要）
        orbital_visualization = {}
        if visualize_orbitals:
            try:
                # 执行最终的HF计算以获得轨道
                final_mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
                qml.qchem.hf_energy(final_mol)()
                
                # 生成轨道数据
                n_grid = 30
                x_range = np.linspace(-3, 3, n_grid)
                y_range = np.linspace(-3, 3, n_grid)
                
                # 原子轨道
                atomic_orb_data = []
                for i in range(len(final_mol.basis_set)):
                    ao = final_mol.atomic_orbital(i)
                    x_grid, y_grid = np.meshgrid(x_range, y_range)
                    ao_values = np.vectorize(ao)(x_grid, y_grid, 0.0)
                    atomic_orb_data.append({
                        "orbital_index": i,
                        "x_grid": x_grid.tolist(),
                        "y_grid": y_grid.tolist(),
                        "values": ao_values.tolist()
                    })
                
                # 分子轨道（如果可用）
                molecular_orb_data = []
                if hasattr(final_mol, 'mo_coefficients'):
                    final_mol.mo_coefficients = final_mol.mo_coefficients.T
                    for i in range(min(2, len(final_mol.mo_coefficients))):  # 只可视化前两个MO
                        mo = final_mol.molecular_orbital(i)
                        mo_values = np.vectorize(mo)(x_grid, y_grid, 0.0)
                        molecular_orb_data.append({
                            "orbital_index": i,
                            "x_grid": x_grid.tolist(),
                            "y_grid": y_grid.tolist(),
                            "values": mo_values.tolist()
                        })
                
                orbital_visualization = {
                    "atomic_orbitals": atomic_orb_data,
                    "molecular_orbitals": molecular_orb_data,
                    "nuclear_positions": geometry.tolist()
                }
                
            except Exception as e:
                orbital_visualization = {
                    "error": f"轨道可视化失败: {str(e)}",
                    "available": False
                }
        
        # 13. 计算最终Hartree-Fock能量作为参考
        try:
            final_mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
            hf_energy = qml.qchem.hf_energy(final_mol)()
        except:
            hf_energy = None
        
        # 14. 构建返回结果
        result = {
            "success": True,
            "molecule": {
                "symbols": symbols,
                "initial_geometry": initial_geometry,
                "n_atoms": n_atoms,
                "n_electrons": electrons
            },
            "final_energy": float(energy_history[-1]),
            "energy_unit": "Hartree",
            "optimized_geometry": geometry.tolist(),
            "optimized_circuit_params": circuit_param.tolist(),
            "optimized_basis": {
                "coefficients": coeff.tolist() if optimize_basis else "not_optimized",
                "exponents": alpha.tolist() if optimize_basis else "not_optimized",
                "basis_optimized": optimize_basis
            },
            "forces": forces.tolist(),
            
            "algorithm_config": full_config,
            
            "optimization_history": {
                "energy_history": energy_history,
                "geometry_history": [g.tolist() for g in geometry_history],
                "circuit_param_history": [p.tolist() for p in circuit_param_history],
                "force_history": [f.tolist() for f in force_history],
                "converged": converged,
                "iterations_performed": len(energy_history),
                "max_iterations_allowed": max_iterations,
                "final_max_force": float(jnp.abs(forces).max()),
                "energy_improvement": energy_history[0] - energy_history[-1] if len(energy_history) > 1 else 0.0
            },
            
            "orbital_visualization": orbital_visualization,
            
            "computational_details": {
                "qubits_used": qubits,
                "optimization_types": {
                    "geometry": optimize_geometry,
                    "circuit": True,
                    "basis_set": optimize_basis
                },
                "learning_rates": {
                    "geometry": geometry_lr,
                    "circuit": circuit_lr,
                    "basis": basis_lr
                },
                "execution_time": end_time - start_time,
                "automatic_differentiation": True,
                "hartree_fock_reference": float(hf_energy) if hf_energy is not None else None,
                "energy_vs_hf": float(energy_history[-1] - hf_energy) if hf_energy is not None else None
            }
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"可微分Hartree-Fock VQE计算过程中发生错误: {str(e)}",
            "algorithm_config": full_config,
            "suggested_action": "请检查JAX环境或参数设置"
        }


def validate_differentiable_hf_vqe_config(config):
    """
    验证可微分Hartree-Fock VQE配置参数的辅助函数
    
    Args:
        config (dict): 配置字典
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(config, dict):
        return False, "配置必须是字典格式"
    
    if 'symbols' not in config or 'initial_geometry' not in config:
        return False, "缺少必需参数: symbols 和 initial_geometry"
    
    symbols = config['symbols']
    geometry = config['initial_geometry']
    
    if not isinstance(symbols, list) or not symbols:
        return False, "symbols必须是非空列表"
    
    if not isinstance(geometry, list) or len(geometry) != len(symbols):
        return False, "geometry长度必须等于原子数量"
    
    # 检查坐标格式
    for coord in geometry:
        if not isinstance(coord, list) or len(coord) != 3:
            return False, "每个原子坐标必须是长度为3的列表[x, y, z]"
    
    user_config = config.get('config', {})
    
    # 验证学习率
    for lr_name in ['geometry_lr', 'circuit_lr', 'basis_lr']:
        if lr_name in user_config:
            lr = user_config[lr_name]
            if not isinstance(lr, (int, float)) or lr <= 0:
                return False, f"{lr_name}必须是正数"
    
    return True, ""


def estimate_differentiable_hf_vqe_resources(symbols, optimize_basis=False):
    """
    估算可微分Hartree-Fock VQE算法所需资源
    
    Args:
        symbols (list): 原子符号列表
        optimize_basis (bool): 是否优化基组参数
        
    Returns:
        dict: 资源估算结果
    """
    try:
        n_atoms = len(symbols)
        
        # 估算电子数和量子比特数
        electron_count = sum([_get_atomic_number(symbol) for symbol in symbols])
        estimated_qubits = n_atoms * 2  # STO-3G每个原子一个基函数
        
        # 估算参数数量
        n_circuit_params = min(electron_count, 5)  # 经验估算
        n_geom_params = n_atoms * 3
        n_basis_params = n_atoms * 6 if optimize_basis else 0  # 3个指数 + 3个系数
        
        total_params = n_circuit_params + n_geom_params + n_basis_params
        
        # 估算计算时间
        time_per_iteration = 0.1 * total_params * estimated_qubits
        estimated_total_time = time_per_iteration * 50  # 假设50次迭代
        
        return {
            "n_atoms": n_atoms,
            "electrons": electron_count,
            "qubits_required": estimated_qubits,
            "feasible": estimated_qubits <= 8,
            "parameter_counts": {
                "circuit": n_circuit_params,
                "geometry": n_geom_params,
                "basis": n_basis_params,
                "total": total_params
            },
            "optimization_features": {
                "simultaneous_optimization": True,
                "automatic_differentiation": True,
                "force_calculation": True,
                "basis_optimization": optimize_basis
            },
            "estimated_time": f"{estimated_total_time:.1f}秒",
            "memory_requirement": "中等",
            "computational_advantages": [
                "精确梯度计算",
                "多参数联合优化", 
                "可获得比标准基组更低能量"
            ]
        }
    except Exception as e:
        return {
            "feasible": False,
            "error": f"无法估算资源: {str(e)}"
        }


def _get_atomic_number(symbol):
    """获取原子序数"""
    atomic_numbers = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10
    }
    return atomic_numbers.get(symbol, 1)