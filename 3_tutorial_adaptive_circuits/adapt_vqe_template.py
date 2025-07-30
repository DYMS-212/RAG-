# adapt_vqe_template.py
import jax
import jax.numpy as jnp
import pennylane as qml
import optax
import numpy as np
import time
from pennylane import qchem

# JAX配置
jax.config.update("jax_enable_x64", True)

def run_adapt_vqe(config):
    """
    ADAPT-VQE算法参数化模板
    
    Args:
        config (dict): 包含symbols, geometry和可选配置参数的字典
            - symbols (list): 原子符号列表
            - geometry (list): 原子几何坐标
            - active_electrons (int, optional): 活跃电子数
            - active_orbitals (int, optional): 活跃轨道数
            - method (str, optional): 构建方法，默认'adaptive_optimizer'
            - gradient_threshold (float, optional): 梯度阈值，默认1e-5
            - max_iterations (int, optional): 最大迭代次数，默认10
            - learning_rate (float, optional): 学习率，默认0.5
            - optimizer (str, optional): 优化器类型，默认'sgd'
            - use_sparse (bool, optional): 是否使用稀疏Hamiltonian，默认False
            - drain_pool (bool, optional): 是否移除已选择的门，默认False
    
    Returns:
        dict: 包含计算结果和完整配置信息的字典
    """
    
    # 1. 提取和验证输入参数
    symbols = config.get('symbols')
    geometry = config.get('geometry')
    
    if not symbols or not geometry:
        return {
            "success": False,
            "error": "缺少必需参数: symbols 和 geometry",
            "suggested_action": "请提供原子符号列表和几何坐标"
        }
    
    # 2. 设置默认参数
    user_config = config.get('config', {})
    active_electrons = user_config.get('active_electrons')
    active_orbitals = user_config.get('active_orbitals')
    method = user_config.get('method', 'adaptive_optimizer')
    gradient_threshold = user_config.get('gradient_threshold', 1e-5)
    max_iterations = user_config.get('max_iterations', 10)
    learning_rate = user_config.get('learning_rate', 0.5)
    optimizer_type = user_config.get('optimizer', 'sgd')
    use_sparse = user_config.get('use_sparse', False)
    drain_pool = user_config.get('drain_pool', False)
    
    # 3. 构建完整配置记录
    full_config = {
        "algorithm_name": "ADAPT-VQE",
        "algorithm_type": "adaptive_variational_quantum_eigensolver_pennylane",
        "parameters_used": {
            "symbols": symbols,
            "geometry": geometry,
            "active_electrons": active_electrons,
            "active_orbitals": active_orbitals,
            "method": method,
            "gradient_threshold": gradient_threshold,
            "max_iterations": max_iterations,
            "learning_rate": learning_rate,
            "optimizer": optimizer_type,
            "use_sparse": use_sparse,
            "drain_pool": drain_pool
        },
        "default_values_applied": {
            "method": method == 'adaptive_optimizer',
            "gradient_threshold": gradient_threshold == 1e-5,
            "max_iterations": max_iterations == 10,
            "learning_rate": learning_rate == 0.5,
            "optimizer": optimizer_type == 'sgd',
            "use_sparse": use_sparse == False,
            "drain_pool": drain_pool == False,
            "active_electrons": active_electrons is None,
            "active_orbitals": active_orbitals is None
        },
        "execution_environment": {
            "device": "lightning.qubit",
            "interface": "jax",
            "differentiation": "parameter-shift" if use_sparse else "auto"
        }
    }
    
    try:
        # 4. 构建分子和Hamiltonian
        try:
            geometry_array = jnp.array(geometry)
            molecule = qchem.Molecule(symbols, geometry_array)
            
            if active_electrons is not None and active_orbitals is not None:
                H, qubits = qchem.molecular_hamiltonian(
                    molecule,
                    active_electrons=active_electrons,
                    active_orbitals=active_orbitals
                )
            else:
                H, qubits = qchem.molecular_hamiltonian(molecule)
                
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
                "error": f"分子需要{qubits}量子比特，超过ADAPT-VQE限制(16个)",
                "algorithm_config": full_config,
                "estimated_qubits": qubits,
                "suggested_action": "请减少活跃轨道数或使用更简单的分子"
            }
        
        # 6. 更新实际使用的参数
        if active_electrons is None:
            active_electrons = 2  # 根据分子自动确定
            for symbol in symbols:
                if symbol == 'Li':
                    active_electrons = 2
                    break
                elif symbol == 'Be':
                    active_electrons = 4
                    break
        
        if active_orbitals is None:
            active_orbitals = 5  # 默认值
        
        full_config["parameters_used"]["active_electrons"] = active_electrons
        full_config["parameters_used"]["active_orbitals"] = active_orbitals
        
        # 7. 生成激发态和HF态
        singles, doubles = qchem.excitations(active_electrons, qubits)
        hf_state = qchem.hf_state(active_electrons, qubits)
        
        # 8. 创建设备
        dev = qml.device("lightning.qubit", wires=qubits)
        
        # 9. 根据方法选择执行策略
        if method == 'adaptive_optimizer':
            result = _run_adaptive_optimizer_method(
                H, qubits, singles, doubles, hf_state, dev,
                max_iterations, learning_rate, use_sparse, drain_pool,
                full_config
            )
        else:  # manual method
            result = _run_manual_method(
                H, qubits, singles, doubles, hf_state, dev,
                gradient_threshold, max_iterations, learning_rate, 
                optimizer_type, use_sparse, full_config
            )
        
        # 10. 添加通用信息
        result.update({
            "molecule": {
                "symbols": symbols,
                "geometry": geometry,
                "active_electrons": active_electrons,
                "active_orbitals": active_orbitals
            },
            "system_info": {
                "qubits_used": qubits,
                "active_electrons": active_electrons,
                "active_orbitals": active_orbitals,
                "total_excitations": len(singles) + len(doubles),
                "single_excitations": len(singles),
                "double_excitations": len(doubles),
                "hartree_fock_state": hf_state.tolist()
            },
            "computational_details": {
                "device_type": "lightning.qubit",
                "backend": "jax",
                "precision": "float64",
                "sparse_optimization": use_sparse
            }
        })
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"ADAPT-VQE计算过程中发生错误: {str(e)}",
            "molecule": {"symbols": symbols, "geometry": geometry},
            "algorithm_config": full_config,
            "suggested_action": "请检查输入参数或联系技术支持"
        }


def _run_adaptive_optimizer_method(H, qubits, singles, doubles, hf_state, dev,
                                 max_iterations, learning_rate, use_sparse, drain_pool,
                                 full_config):
    """使用PennyLane AdaptiveOptimizer的方法"""
    
    # 创建操作池
    singles_excitations = [qml.SingleExcitation(0.0, x) for x in singles]
    doubles_excitations = [qml.DoubleExcitation(0.0, x) for x in doubles]
    operator_pool = doubles_excitations + singles_excitations
    
    # 定义初始电路
    @qml.qnode(dev)
    def circuit():
        [qml.PauliX(i) for i in np.nonzero(hf_state)[0]]
        return qml.expval(H)
    
    # 运行自适应优化
    opt = qml.optimize.AdaptiveOptimizer()
    circuit_history = []
    energy_history = []
    gradient_history = []
    
    start_time = time.time()
    
    for i in range(len(operator_pool)):
        circuit, energy, gradient = opt.step_and_cost(circuit, operator_pool, drain_pool=drain_pool)
        
        circuit_history.append(str(qml.draw(circuit, decimals=None)()))
        energy_history.append(float(energy))
        gradient_history.append(float(gradient))
        
        if gradient < 3e-3:  # 收敛条件
            break
    
    end_time = time.time()
    
    return {
        "success": True,
        "ground_state_energy": energy_history[-1] if energy_history else 0.0,
        "energy_unit": "Hartree",
        "algorithm_config": full_config,
        "circuit_construction": {
            "method": "adaptive_optimizer",
            "total_steps": len(energy_history),
            "final_gradient": gradient_history[-1] if gradient_history else 0.0,
            "converged": gradient_history[-1] < 3e-3 if gradient_history else False,
            "drain_pool_used": drain_pool
        },
        "optimization_info": {
            "energy_history": energy_history,
            "gradient_history": gradient_history,
            "execution_time": end_time - start_time,
            "convergence_threshold": 3e-3
        }
    }


def _run_manual_method(H, qubits, singles, doubles, hf_state, dev,
                      gradient_threshold, max_iterations, learning_rate,
                      optimizer_type, use_sparse, full_config):
    """手动分组选择的方法"""
    
    start_time = time.time()
    
    # 阶段1：选择重要的双激发
    def circuit_1(params, excitations):
        qml.BasisState(jnp.array(hf_state), wires=range(qubits))
        for i, excitation in enumerate(excitations):
            if len(excitation) == 4:
                qml.DoubleExcitation(params[i], wires=excitation)
            else:
                qml.SingleExcitation(params[i], wires=excitation)
        return qml.expval(H)
    
    cost_fn = qml.QNode(circuit_1, dev, interface="jax")
    circuit_gradient = jax.grad(cost_fn, argnums=0)
    
    # 计算双激发梯度
    params = [0.0] * len(doubles)
    grads = circuit_gradient(params, excitations=doubles)
    
    # 选择重要的双激发
    doubles_select = [doubles[i] for i in range(len(doubles)) if abs(grads[i]) > gradient_threshold]
    double_gradients = [float(grads[i]) for i in range(len(doubles)) if abs(grads[i]) > gradient_threshold]
    
    # 阶段2：优化选定的双激发
    if doubles_select:
        params_doubles = jnp.zeros(len(doubles_select))
        
        # 选择优化器
        if optimizer_type == 'sgd':
            opt = optax.sgd(learning_rate=learning_rate)
        elif optimizer_type == 'adam':
            opt = optax.adam(learning_rate=learning_rate)
        elif optimizer_type == 'adagrad':
            opt = optax.adagrad(learning_rate=learning_rate)
        else:
            opt = optax.sgd(learning_rate=learning_rate)
        
        opt_state = opt.init(params_doubles)
        
        for _ in range(max_iterations):
            gradient = jax.grad(cost_fn, argnums=0)(params_doubles, excitations=doubles_select)
            updates, opt_state = opt.update(gradient, opt_state)
            params_doubles = optax.apply_updates(params_doubles, updates)
    else:
        params_doubles = jnp.array([])
    
    # 阶段3：选择重要的单激发
    def circuit_2(params, excitations, gates_select, params_select):
        qml.BasisState(hf_state, wires=range(qubits))
        
        for i, gate in enumerate(gates_select):
            if len(gate) == 4:
                qml.DoubleExcitation(params_select[i], wires=gate)
            elif len(gate) == 2:
                qml.SingleExcitation(params_select[i], wires=gate)
        
        for i, gate in enumerate(excitations):
            if len(gate) == 4:
                qml.DoubleExcitation(params[i], wires=gate)
            elif len(gate) == 2:
                qml.SingleExcitation(params[i], wires=gate)
        
        return qml.expval(H)
    
    cost_fn = qml.QNode(circuit_2, dev, interface="jax")
    circuit_gradient = jax.grad(cost_fn, argnums=0)
    params = [0.0] * len(singles)
    
    grads = circuit_gradient(
        params,
        excitations=singles,
        gates_select=doubles_select,
        params_select=params_doubles
    )
    
    # 选择重要的单激发
    singles_select = [singles[i] for i in range(len(singles)) if abs(grads[i]) > gradient_threshold]
    single_gradients = [float(grads[i]) for i in range(len(singles)) if abs(grads[i]) > gradient_threshold]
    
    # 阶段4：最终优化
    gates_select = doubles_select + singles_select
    
    if gates_select:
        if use_sparse:
            # 使用稀疏Hamiltonian
            H_sparse = H.sparse_matrix()
            
            @qml.qnode(dev, diff_method="parameter-shift", interface="jax")
            def final_circuit(params):
                qml.BasisState(hf_state, wires=range(qubits))
                
                for i, excitation in enumerate(gates_select):
                    if len(excitation) == 4:
                        qml.DoubleExcitation(params[i], wires=excitation)
                    elif len(excitation) == 2:
                        qml.SingleExcitation(params[i], wires=excitation)
                
                return qml.expval(qml.SparseHamiltonian(H_sparse, wires=range(qubits)))
        else:
            # 使用常规Hamiltonian
            final_circuit = qml.QNode(circuit_1, dev, interface="jax")
        
        params = jnp.zeros(len(gates_select))
        
        # 选择优化器
        if optimizer_type == 'sgd':
            opt = optax.sgd(learning_rate=learning_rate)
        elif optimizer_type == 'adam':
            opt = optax.adam(learning_rate=learning_rate)
        elif optimizer_type == 'adagrad':
            opt = optax.adagrad(learning_rate=learning_rate)
        else:
            opt = optax.sgd(learning_rate=learning_rate)
        
        opt_state = opt.init(params)
        energy_history = []
        
        for n in range(max_iterations):
            if use_sparse:
                energy = final_circuit(params)
                gradient = jax.grad(final_circuit, argnums=0)(params)
            else:
                energy = final_circuit(params, excitations=gates_select)
                gradient = jax.grad(final_circuit, argnums=0)(params, excitations=gates_select)
            
            energy_history.append(float(energy))
            updates, opt_state = opt.update(gradient, opt_state)
            params = optax.apply_updates(params, updates)
        
        final_energy = energy_history[-1] if energy_history else 0.0
    else:
        # 没有选中任何门，使用HF能量
        @qml.qnode(dev)
        def hf_circuit():
            [qml.PauliX(i) for i in np.nonzero(hf_state)[0]]
            return qml.expval(H)
        final_energy = float(hf_circuit())
        energy_history = [final_energy]
    
    end_time = time.time()
    
    return {
        "success": True,
        "ground_state_energy": final_energy,
        "energy_unit": "Hartree",
        "algorithm_config": full_config,
        "circuit_construction": {
            "method": "manual",
            "selected_doubles": len(doubles_select),
            "selected_singles": len(singles_select),
            "total_selected": len(gates_select),
            "total_available": len(singles) + len(doubles),
            "gradient_threshold": gradient_threshold,
            "double_gradients": {
                "selected": double_gradients,
                "max_gradient": max(abs(g) for g in double_gradients) if double_gradients else 0.0
            },
            "single_gradients": {
                "selected": single_gradients,
                "max_gradient": max(abs(g) for g in single_gradients) if single_gradients else 0.0
            }
        },
        "optimization_info": {
            "energy_history": energy_history,
            "max_iterations_per_stage": max_iterations,
            "execution_time": end_time - start_time,
            "sparse_used": use_sparse,
            "final_parameters": params.tolist() if len(gates_select) > 0 else []
        }
    }


def validate_adapt_vqe_config(config):
    """
    验证ADAPT-VQE配置参数的辅助函数
    
    Args:
        config (dict): 配置字典
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(config, dict):
        return False, "配置必须是字典格式"
    
    if 'symbols' not in config or 'geometry' not in config:
        return False, "缺少必需参数: symbols 和 geometry"
    
    symbols = config['symbols']
    geometry = config['geometry']
    
    if not isinstance(symbols, list) or not symbols:
        return False, "symbols必须是非空列表"
    
    if not isinstance(geometry, list) or len(geometry) != len(symbols):
        return False, "geometry长度必须等于原子数量"
    
    user_config = config.get('config', {})
    
    # 验证method参数
    if 'method' in user_config:
        if user_config['method'] not in ['adaptive_optimizer', 'manual']:
            return False, "method必须是'adaptive_optimizer'或'manual'"
    
    return True, ""


def estimate_adapt_vqe_resources(symbols, geometry, active_electrons=None, active_orbitals=None):
    """
    估算ADAPT-VQE算法所需资源
    
    Args:
        symbols (list): 原子符号列表
        geometry (list): 原子几何坐标
        active_electrons (int, optional): 活跃电子数
        active_orbitals (int, optional): 活跃轨道数
        
    Returns:
        dict: 资源估算结果
    """
    try:
        geometry_array = jnp.array(geometry)
        molecule = qchem.Molecule(symbols, geometry_array)
        
        if active_electrons is not None and active_orbitals is not None:
            H, qubits = qchem.molecular_hamiltonian(
                molecule,
                active_electrons=active_electrons,
                active_orbitals=active_orbitals
            )
        else:
            H, qubits = qchem.molecular_hamiltonian(molecule)
        
        # 估算激发数量
        estimated_electrons = active_electrons if active_electrons else 2
        singles, doubles = qchem.excitations(estimated_electrons, qubits)
        
        # 估算稀疏度
        H_sparse = H.sparse_matrix()
        sparsity = H_sparse.nnz / (H_sparse.shape[0] * H_sparse.shape[1])
        
        return {
            "qubits_required": qubits,
            "total_excitations": len(singles) + len(doubles),
            "single_excitations": len(singles),
            "double_excitations": len(doubles),
            "hamiltonian_sparsity": f"{sparsity:.6f}",
            "feasible": qubits <= 16,
            "estimated_selected_gates": "基于梯度自适应确定",
            "sparse_acceleration": f"约{1/sparsity:.1f}x加速" if sparsity > 0 else "unknown"
        }
    except Exception as e:
        return {
            "qubits_required": "unknown",
            "total_excitations": "unknown",
            "feasible": False,
            "error": f"无法分析分子: {str(e)}"
        }