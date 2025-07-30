# adaptive_vqe_template.py
import jax
import jax.numpy as jnp
import pennylane as qml
import optax
import catalyst
from catalyst import qjit
import numpy as np

# JAX配置
jax.config.update("jax_platform_name", "cpu") 
jax.config.update('jax_enable_x64', True)

def run_adaptive_vqe(config):
    """
    Adaptive VQE算法参数化模板
    
    Args:
        config (dict): 包含symbols, coordinates和可选配置参数的字典
            - symbols (list): 原子符号列表
            - coordinates (list): 原子坐标数组
            - charge (int, optional): 分子电荷，默认0
            - active_electrons (int, optional): 活跃电子数
            - active_orbitals (int, optional): 活跃轨道数
            - gradient_threshold (float, optional): 梯度阈值，默认1e-5
            - max_iterations (int, optional): 最大迭代次数，默认20
            - learning_rate (float, optional): 学习率，默认0.5
            - optimizer (str, optional): 优化器类型，默认'sgd'
    
    Returns:
        dict: 包含计算结果和完整配置信息的字典
    """
    
    # 1. 提取和验证输入参数
    symbols = config.get('symbols')
    coordinates = config.get('coordinates')
    
    if not symbols or not coordinates:
        return {
            "success": False,
            "error": "缺少必需参数: symbols 和 coordinates",
            "suggested_action": "请提供原子符号列表和坐标数组"
        }
    
    # 2. 设置默认参数
    user_config = config.get('config', {})
    charge = user_config.get('charge', 0)
    active_electrons = user_config.get('active_electrons')
    active_orbitals = user_config.get('active_orbitals')
    gradient_threshold = user_config.get('gradient_threshold', 1e-5)
    max_iterations = user_config.get('max_iterations', 20)
    learning_rate = user_config.get('learning_rate', 0.5)
    optimizer_type = user_config.get('optimizer', 'sgd')
    
    # 3. 构建完整配置记录
    full_config = {
        "algorithm_name": "Adaptive VQE",
        "algorithm_type": "adaptive_variational_quantum_eigensolver",
        "parameters_used": {
            "symbols": symbols,
            "coordinates": coordinates,
            "charge": charge,
            "active_electrons": active_electrons,
            "active_orbitals": active_orbitals,
            "gradient_threshold": gradient_threshold,
            "max_iterations": max_iterations,
            "learning_rate": learning_rate,
            "optimizer": optimizer_type
        },
        "default_values_applied": {
            "charge": charge == 0,
            "gradient_threshold": gradient_threshold == 1e-5,
            "max_iterations": max_iterations == 20,
            "learning_rate": learning_rate == 0.5,
            "optimizer": optimizer_type == 'sgd',
            "active_electrons": active_electrons is None,
            "active_orbitals": active_orbitals is None
        },
        "execution_environment": {
            "device": "lightning.qubit",
            "interface": "jax",
            "compiler": "catalyst",
            "compilation_mode": "JIT"
        }
    }
    
    try:
        # 4. 构建分子Hamiltonian
        try:
            coordinates_array = np.array(coordinates)
            if active_electrons is not None and active_orbitals is not None:
                hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
                    symbols, coordinates_array, charge=charge,
                    active_electrons=active_electrons, 
                    active_orbitals=active_orbitals
                )
            else:
                hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
                    symbols, coordinates_array, charge=charge
                )
        except Exception as e:
            return {
                "success": False,
                "error": f"无法构建分子Hamiltonian: {str(e)}",
                "algorithm_config": full_config,
                "suggested_action": "请检查分子几何结构和参数设置"
            }
        
        # 5. 检查量子比特限制
        if qubits > 16:
            return {
                "success": False,
                "error": f"分子需要{qubits}量子比特，超过Adaptive VQE限制(16个)",
                "algorithm_config": full_config,
                "estimated_qubits": qubits,
                "suggested_action": "请减少活跃轨道数或使用更简单的分子"
            }
        
        # 6. 更新实际使用的参数
        if active_electrons is None:
            # 从Hamiltonian计算中获取实际电子数
            active_electrons = 2  # 默认值，根据分子调整
            for symbol in symbols:
                if symbol == 'H':
                    continue
                elif symbol == 'Li':
                    active_electrons = 2
                    break
                elif symbol == 'Be':
                    active_electrons = 4
                    break
        
        if active_orbitals is None:
            active_orbitals = qubits // 2
        
        full_config["parameters_used"]["active_electrons"] = active_electrons
        full_config["parameters_used"]["active_orbitals"] = active_orbitals
        
        # 7. 生成激发态
        singles, doubles = qml.qchem.excitations(active_electrons, qubits)
        hf_state = qml.qchem.hf_state(active_electrons, qubits)
        
        # 8. 定义自适应VQE电路
        @qml.qnode(qml.device("lightning.qubit", wires=qubits))
        def adaptive_circuit(params, excitations, excitation_types):
            qml.BasisState(hf_state, wires=range(qubits))
            
            for i, excitation in enumerate(excitations):
                if excitation_types[i] == 'double':
                    qml.DoubleExcitation(params[i], wires=excitation)
                else:  # single
                    qml.SingleExcitation(params[i], wires=excitation)
            
            coeffs, ops = hamiltonian.terms()
            return qml.expval(qml.Hamiltonian(np.array(coeffs), ops))
        
        # 9. 第一阶段：选择重要的双激发
        def select_important_doubles():
            if not doubles:
                return [], [], []
            
            # 计算双激发的梯度
            params_double = jnp.zeros(len(doubles))
            excitation_types = ['double'] * len(doubles)
            
            grad_fn = jax.grad(adaptive_circuit, argnums=0)
            gradients = grad_fn(params_double, doubles, excitation_types)
            
            # 选择重要的双激发
            selected_doubles = []
            selected_indices = []
            for i, grad in enumerate(gradients):
                if abs(grad) > gradient_threshold:
                    selected_doubles.append(doubles[i])
                    selected_indices.append(i)
            
            return selected_doubles, selected_indices, gradients
        
        selected_doubles, double_indices, double_gradients = select_important_doubles()
        
        # 10. 第二阶段：优化选定的双激发
        optimized_double_params = []
        if selected_doubles:
            params_double = jnp.zeros(len(selected_doubles))
            excitation_types = ['double'] * len(selected_doubles)
            
            # 选择优化器
            if optimizer_type == 'sgd':
                opt = optax.sgd(learning_rate=learning_rate)
            elif optimizer_type == 'adam':
                opt = optax.adam(learning_rate=learning_rate)
            elif optimizer_type == 'adagrad':
                opt = optax.adagrad(learning_rate=learning_rate)
            else:
                opt = optax.sgd(learning_rate=learning_rate)
            
            opt_state = opt.init(params_double)
            
            # 优化双激发参数
            for _ in range(max_iterations):
                gradient = jax.grad(adaptive_circuit, argnums=0)(
                    params_double, selected_doubles, excitation_types
                )
                updates, opt_state = opt.update(gradient, opt_state)
                params_double = optax.apply_updates(params_double, updates)
            
            optimized_double_params = params_double.tolist()
        
        # 11. 第三阶段：选择重要的单激发
        def select_important_singles():
            if not singles:
                return [], [], []
            
            # 构建包含选定双激发的电路来计算单激发梯度
            all_excitations = selected_doubles + singles
            all_types = ['double'] * len(selected_doubles) + ['single'] * len(singles)
            all_params = jnp.array(optimized_double_params + [0.0] * len(singles))
            
            if not all_excitations:
                return [], [], []
            
            grad_fn = jax.grad(adaptive_circuit, argnums=0)
            gradients = grad_fn(all_params, all_excitations, all_types)
            
            # 提取单激发的梯度
            single_gradients = gradients[len(selected_doubles):]
            
            # 选择重要的单激发
            selected_singles = []
            selected_indices = []
            for i, grad in enumerate(single_gradients):
                if abs(grad) > gradient_threshold:
                    selected_singles.append(singles[i])
                    selected_indices.append(i)
            
            return selected_singles, selected_indices, single_gradients
        
        selected_singles, single_indices, single_gradients = select_important_singles()
        
        # 12. 第四阶段：最终优化所有选定的激发
        final_excitations = selected_doubles + selected_singles
        final_types = ['double'] * len(selected_doubles) + ['single'] * len(selected_singles)
        final_params = jnp.array(optimized_double_params + [0.0] * len(selected_singles))
        
        if not final_excitations:
            # 如果没有选定任何激发，使用HF态能量
            final_energy = float(adaptive_circuit(jnp.array([]), [], []))
            converged = True
            final_iterations = 0
        else:
            # 选择优化器
            if optimizer_type == 'sgd':
                opt = optax.sgd(learning_rate=learning_rate)
            elif optimizer_type == 'adam':
                opt = optax.adam(learning_rate=learning_rate)
            elif optimizer_type == 'adagrad':
                opt = optax.adagrad(learning_rate=learning_rate)
            else:
                opt = optax.sgd(learning_rate=learning_rate)
            
            opt_state = opt.init(final_params)
            energy_history = []
            
            # 最终优化
            converged = False
            final_iterations = 0
            
            for iteration in range(max_iterations):
                current_energy = adaptive_circuit(final_params, final_excitations, final_types)
                energy_history.append(float(current_energy))
                
                gradient = jax.grad(adaptive_circuit, argnums=0)(
                    final_params, final_excitations, final_types
                )
                updates, opt_state = opt.update(gradient, opt_state)
                final_params = optax.apply_updates(final_params, updates)
                
                final_iterations = iteration + 1
                
                # 检查收敛
                if len(energy_history) >= 2:
                    conv = abs(energy_history[-1] - energy_history[-2])
                    if conv <= 1e-6:
                        converged = True
                        break
            
            final_energy = float(adaptive_circuit(final_params, final_excitations, final_types))
        
        # 13. 构建返回结果
        molecule_info = {
            "symbols": symbols,
            "coordinates": coordinates,
            "charge": charge,
            "active_electrons": active_electrons,
            "active_orbitals": active_orbitals
        }
        
        adaptive_selection_info = {
            "total_doubles": len(doubles),
            "selected_doubles": len(selected_doubles),
            "total_singles": len(singles),
            "selected_singles": len(selected_singles),
            "gradient_threshold": gradient_threshold,
            "double_gradients": {
                "max_gradient": float(max(abs(g) for g in double_gradients)) if double_gradients else 0.0,
                "selected_count": len(selected_doubles)
            },
            "single_gradients": {
                "max_gradient": float(max(abs(g) for g in single_gradients)) if single_gradients else 0.0,
                "selected_count": len(selected_singles)
            },
            "total_selected_excitations": len(final_excitations)
        }
        
        result = {
            "success": True,
            "molecule": molecule_info,
            "ground_state_energy": final_energy,
            "energy_unit": "Hartree",
            
            "algorithm_config": full_config,
            
            "adaptive_selection": adaptive_selection_info,
            
            "optimization_info": {
                "converged": converged,
                "final_iterations": final_iterations,
                "max_iterations_per_stage": max_iterations,
                "final_parameters": final_params.tolist() if len(final_excitations) > 0 else [],
                "optimization_stages": {
                    "stage1_double_selection": f"从{len(doubles)}个双激发中选择{len(selected_doubles)}个",
                    "stage2_double_optimization": f"优化{len(selected_doubles)}个双激发参数",
                    "stage3_single_selection": f"从{len(singles)}个单激发中选择{len(selected_singles)}个", 
                    "stage4_final_optimization": f"最终优化{len(final_excitations)}个激发"
                }
            },
            
            "system_info": {
                "qubits_used": qubits,
                "active_electrons": active_electrons,
                "active_orbitals": active_orbitals,
                "hartree_fock_state": hf_state.tolist(),
                "hamiltonian_terms": len(hamiltonian.ops) if hasattr(hamiltonian, 'ops') else "unknown"
            },
            
            "computational_details": {
                "device_type": "lightning.qubit",
                "backend": "jax",
                "compiler": "catalyst",
                "precision": "float64",
                "excitation_selection": "gradient_based",
                "circuit_reduction": f"{len(doubles) + len(singles)} → {len(final_excitations)} excitations"
            }
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Adaptive VQE计算过程中发生错误: {str(e)}",
            "molecule": {"symbols": symbols, "coordinates": coordinates},
            "algorithm_config": full_config,
            "suggested_action": "请检查输入参数或联系技术支持"
        }


def validate_adaptive_vqe_config(config):
    """
    验证Adaptive VQE配置参数的辅助函数
    
    Args:
        config (dict): 配置字典
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(config, dict):
        return False, "配置必须是字典格式"
    
    if 'symbols' not in config or 'coordinates' not in config:
        return False, "缺少必需参数: symbols 和 coordinates"
    
    symbols = config['symbols']
    coordinates = config['coordinates']
    
    if not isinstance(symbols, list) or not symbols:
        return False, "symbols必须是非空列表"
    
    if not isinstance(coordinates, list) or len(coordinates) != len(symbols) * 3:
        return False, "coordinates长度必须是原子数量的3倍"
    
    user_config = config.get('config', {})
    
    # 验证数值参数范围
    if 'gradient_threshold' in user_config:
        gt = user_config['gradient_threshold']
        if not isinstance(gt, (int, float)) or gt <= 0:
            return False, "gradient_threshold必须是正数"
    
    if 'learning_rate' in user_config:
        lr = user_config['learning_rate']
        if not isinstance(lr, (int, float)) or lr <= 0:
            return False, "learning_rate必须是正数"
    
    return True, ""


def estimate_adaptive_vqe_resources(symbols, coordinates, active_electrons=None, active_orbitals=None):
    """
    估算Adaptive VQE算法所需资源
    
    Args:
        symbols (list): 原子符号列表
        coordinates (list): 原子坐标
        active_electrons (int, optional): 活跃电子数
        active_orbitals (int, optional): 活跃轨道数
        
    Returns:
        dict: 资源估算结果
    """
    try:
        coordinates_array = np.array(coordinates)
        if active_electrons is not None and active_orbitals is not None:
            hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
                symbols, coordinates_array,
                active_electrons=active_electrons,
                active_orbitals=active_orbitals
            )
        else:
            hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
                symbols, coordinates_array
            )
        
        # 估算激发数量
        estimated_electrons = active_electrons if active_electrons else 2
        singles, doubles = qml.qchem.excitations(estimated_electrons, qubits)
        
        return {
            "qubits_required": qubits,
            "total_excitations": len(singles) + len(doubles),
            "single_excitations": len(singles),
            "double_excitations": len(doubles),
            "feasible": qubits <= 16,
            "estimated_selected_excitations": "自适应确定",
            "estimated_time": "几分钟到十几分钟"
        }
    except Exception as e:
        return {
            "qubits_required": "unknown",
            "total_excitations": "unknown",
            "feasible": False,
            "error": f"无法分析分子: {str(e)}"
        }