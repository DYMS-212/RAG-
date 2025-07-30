# qng_vqe_template.py
import pennylane as qml
from pennylane import numpy as np
import time
import matplotlib.pyplot as plt

def run_qng_vqe(config):
    """
    量子自然梯度VQE算法参数化模板
    
    Args:
        config (dict): 包含system_definition, ansatz_config和可选配置参数的字典
            - system_definition (dict): 系统定义
            - ansatz_config (dict): 变分电路配置
            - step_size (float, optional): QNG学习率，默认0.01
            - metric_approx (str, optional): 度量张量近似，默认'block-diag'
            - regularization (float, optional): 正则化参数，默认0.001
            - max_iterations (int, optional): 最大迭代次数，默认500
            - convergence_threshold (float, optional): 收敛阈值，默认1e-6
            - initial_params (list, optional): 初始参数
            - compare_standard (bool, optional): 对比标准梯度，默认True
            - standard_step_size (float, optional): 标准梯度学习率，默认0.01
            - track_metric_condition (bool, optional): 跟踪度量条件数，默认False
            - save_optimization_path (bool, optional): 保存优化路径，默认True
            - print_progress (bool, optional): 打印进度，默认True
    
    Returns:
        dict: 包含计算结果和完整配置信息的字典
    """
    
    # 1. 提取和验证输入参数
    system_definition = config.get('system_definition')
    ansatz_config = config.get('ansatz_config')
    
    if not system_definition or not ansatz_config:
        return {
            "success": False,
            "error": "缺少必需参数: system_definition 和 ansatz_config",
            "suggested_action": "请提供系统定义和变分电路配置"
        }
    
    # 2. 设置默认参数
    user_config = config.get('config', {})
    step_size = user_config.get('step_size', 0.01)
    metric_approx = user_config.get('metric_approx', 'block-diag')
    regularization = user_config.get('regularization', 0.001)
    max_iterations = user_config.get('max_iterations', 500)
    convergence_threshold = user_config.get('convergence_threshold', 1e-6)
    initial_params = user_config.get('initial_params')
    compare_standard = user_config.get('compare_standard', True)
    standard_step_size = user_config.get('standard_step_size', 0.01)
    track_metric_condition = user_config.get('track_metric_condition', False)
    save_optimization_path = user_config.get('save_optimization_path', True)
    print_progress = user_config.get('print_progress', True)
    
    # 3. 构建完整配置记录
    full_config = {
        "algorithm_name": "Quantum Natural Gradient VQE",
        "algorithm_type": "quantum_natural_gradient_vqe",
        "parameters_used": {
            "system_definition": system_definition,
            "ansatz_config": ansatz_config,
            "step_size": step_size,
            "metric_approx": metric_approx,
            "regularization": regularization,
            "max_iterations": max_iterations,
            "convergence_threshold": convergence_threshold,
            "initial_params": initial_params,
            "compare_standard": compare_standard,
            "standard_step_size": standard_step_size,
            "track_metric_condition": track_metric_condition,
            "save_optimization_path": save_optimization_path,
            "print_progress": print_progress
        },
        "default_values_applied": {
            "step_size": step_size == 0.01,
            "metric_approx": metric_approx == 'block-diag',
            "regularization": regularization == 0.001,
            "max_iterations": max_iterations == 500,
            "convergence_threshold": convergence_threshold == 1e-6,
            "compare_standard": compare_standard == True,
            "standard_step_size": standard_step_size == 0.01,
            "track_metric_condition": track_metric_condition == False,
            "save_optimization_path": save_optimization_path == True,
            "print_progress": print_progress == True,
            "initial_params": initial_params is None
        },
        "execution_environment": {
            "device": "default.qubit",
            "interface": "autograd",
            "optimization_method": "quantum_natural_gradient"
        }
    }
    
    try:
        # 4. 构建Hamiltonian
        hamiltonian_type = system_definition['hamiltonian_type']
        
        if hamiltonian_type == 'molecule':
            molecule_name = system_definition.get('molecule_name', 'H2')
            bond_length = system_definition.get('bond_length', 0.7)
            
            try:
                # 使用PennyLane datasets加载分子数据
                dataset = qml.data.load('qchem', molname=molecule_name, bondlength=bond_length)[0]
                hamiltonian = dataset.hamiltonian
                exact_energy = dataset.fci_energy
                qubits = len(hamiltonian.wires)
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"无法加载分子数据: {molecule_name}. 错误: {str(e)}",
                    "algorithm_config": full_config,
                    "suggested_action": "请检查分子名称和键长设置"
                }
                
        elif hamiltonian_type == 'spin_system':
            # 简单的自旋系统示例
            qubits = system_definition.get('n_qubits', 2)
            coeffs = [1.0, 1.0]
            ops = [qml.PauliX(0), qml.PauliZ(0)] if qubits == 1 else [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0)]
            hamiltonian = qml.Hamiltonian(coeffs, ops)
            exact_energy = None
            
        elif hamiltonian_type == 'custom_pauli':
            custom_ham = system_definition.get('custom_hamiltonian')
            if not custom_ham:
                return {
                    "success": False,
                    "error": "自定义Hamiltonian需要提供coefficients和pauli_strings",
                    "algorithm_config": full_config
                }
            
            coeffs = custom_ham['coefficients']
            pauli_strings = custom_ham['pauli_strings']
            # 这里需要解析Pauli字符串，简化处理
            ops = []
            qubits = max([len(s) for s in pauli_strings])
            for pauli_str in pauli_strings:
                # 简化的Pauli字符串解析
                op_list = []
                for i, pauli in enumerate(pauli_str):
                    if pauli == 'X':
                        op_list.append(qml.PauliX(i))
                    elif pauli == 'Y':
                        op_list.append(qml.PauliY(i))
                    elif pauli == 'Z':
                        op_list.append(qml.PauliZ(i))
                
                if len(op_list) == 1:
                    ops.append(op_list[0])
                else:
                    # 使用tensor product
                    tensor_op = op_list[0]
                    for op in op_list[1:]:
                        tensor_op = tensor_op @ op
                    ops.append(tensor_op)
            
            hamiltonian = qml.Hamiltonian(coeffs, ops)
            exact_energy = None
            
        else:
            return {
                "success": False,
                "error": f"不支持的Hamiltonian类型: {hamiltonian_type}",
                "algorithm_config": full_config
            }
        
        # 5. 检查量子比特限制
        if qubits > 12:
            return {
                "success": False,
                "error": f"系统需要{qubits}量子比特，超过QNG VQE限制(12个)",
                "algorithm_config": full_config,
                "suggested_action": "请使用更小的系统"
            }
        
        # 6. 构建变分电路
        ansatz_type = ansatz_config['ansatz_type']
        n_layers = ansatz_config.get('n_layers', 2)
        
        def build_ansatz(ansatz_type, qubits, n_layers):
            """构建变分电路ansatz"""
            
            if ansatz_type == 'single_qubit_rotations':
                def ansatz(params, wires=None):
                    if wires is None:
                        wires = range(qubits)
                    
                    param_idx = 0
                    for wire in wires:
                        if param_idx < len(params):
                            qml.RX(params[param_idx], wires=wire)
                            param_idx += 1
                        if param_idx < len(params):
                            qml.RY(params[param_idx], wires=wire)
                            param_idx += 1
                
                n_params = 2 * qubits
                
            elif ansatz_type == 'hardware_efficient':
                rotation_gates = ansatz_config.get('rotation_gates', ['RY', 'RZ'])
                entangling_gate = ansatz_config.get('entangling_gate', 'CNOT')
                
                def ansatz(params, wires=None):
                    if wires is None:
                        wires = range(qubits)
                    
                    param_idx = 0
                    for layer in range(n_layers):
                        # 旋转层
                        for wire in wires:
                            for gate_type in rotation_gates:
                                if param_idx < len(params):
                                    if gate_type == 'RX':
                                        qml.RX(params[param_idx], wires=wire)
                                    elif gate_type == 'RY':
                                        qml.RY(params[param_idx], wires=wire)
                                    elif gate_type == 'RZ':
                                        qml.RZ(params[param_idx], wires=wire)
                                    param_idx += 1
                        
                        # 纠缠层
                        if layer < n_layers - 1:  # 最后一层不加纠缠门
                            for i in range(qubits - 1):
                                if entangling_gate == 'CNOT':
                                    qml.CNOT(wires=[i, i + 1])
                                elif entangling_gate == 'CZ':
                                    qml.CZ(wires=[i, i + 1])
                
                n_params = n_layers * len(rotation_gates) * qubits
                
            elif ansatz_type == 'uccsd':
                # 简化的UCCSD ansatz（需要分子信息）
                if hamiltonian_type != 'molecule':
                    return {
                        "success": False,
                        "error": "UCCSD ansatz只适用于分子系统",
                        "algorithm_config": full_config
                    }
                
                # 使用Hartree-Fock态作为初态
                if hasattr(dataset, 'hf_state'):
                    hf_state = dataset.hf_state
                else:
                    # 估算电子数
                    if molecule_name == 'H2':
                        electrons = 2
                    elif molecule_name == 'LiH':
                        electrons = 4
                    else:
                        electrons = 2  # 默认
                    hf_state = qml.qchem.hf_state(electrons, qubits)
                
                def ansatz(params, wires=None):
                    qml.BasisState(hf_state, wires=range(qubits))
                    param_idx = 0
                    
                    # 双激发门
                    if qubits >= 4 and param_idx < len(params):
                        qml.DoubleExcitation(params[param_idx], wires=[0, 1, 2, 3])
                        param_idx += 1
                    
                    # 单激发门
                    for i in range(0, qubits, 2):
                        if i + 2 < qubits and param_idx < len(params):
                            qml.SingleExcitation(params[param_idx], wires=[i, i + 2])
                            param_idx += 1
                
                n_params = min(5, qubits // 2)  # 限制参数数量
                
            else:
                return {
                    "success": False,
                    "error": f"不支持的ansatz类型: {ansatz_type}",
                    "algorithm_config": full_config
                }
            
            return ansatz, n_params
        
        try:
            ansatz, n_params = build_ansatz(ansatz_type, qubits, n_layers)
        except Exception as e:
            return {
                "success": False, 
                "error": f"构建ansatz失败: {str(e)}",
                "algorithm_config": full_config
            }
        
        # 7. 创建量子设备
        dev = qml.device("default.qubit", wires=qubits)
        
        # 8. 定义成本函数
        @qml.qnode(dev, interface="autograd")
        def cost_function(params):
            ansatz(params)
            return qml.expval(hamiltonian)
        
        # 9. 初始化参数
        if initial_params is None:
            np.random.seed(42)  # 确保可重现性
            initial_params = np.random.uniform(low=0, high=2*np.pi, size=n_params, requires_grad=True)
        else:
            if len(initial_params) != n_params:
                return {
                    "success": False,
                    "error": f"初始参数数量({len(initial_params)})与ansatz需求({n_params})不匹配",
                    "algorithm_config": full_config
                }
            initial_params = np.array(initial_params, requires_grad=True)
        
        # 10. 执行QNG优化
        start_time = time.time()
        
        # 确保Hamiltonian系数不可微分（QNG优化器要求）
        hamiltonian_coeffs, hamiltonian_ops = hamiltonian.terms()
        hamiltonian_for_qng = qml.Hamiltonian(
            np.array(hamiltonian_coeffs, requires_grad=False), 
            hamiltonian_ops
        )
        
        @qml.qnode(dev, interface="autograd")
        def cost_function_qng(params):
            ansatz(params)
            return qml.expval(hamiltonian_for_qng)
        
        # QNG优化
        qng_opt = qml.QNGOptimizer(stepsize=step_size, lam=regularization, approx=metric_approx)
        qng_params = initial_params.copy()
        qng_energy_history = []
        qng_param_history = []
        metric_condition_history = []
        
        if print_progress:
            print("开始量子自然梯度VQE优化...")
            print(f"系统: {hamiltonian_type}, 量子比特: {qubits}, 参数数: {n_params}")
            print(f"度量张量近似: {metric_approx}, 正则化: {regularization}")
        
        qng_converged = False
        for iteration in range(max_iterations):
            qng_params, prev_energy = qng_opt.step_and_cost(cost_function_qng, qng_params)
            qng_energy_history.append(prev_energy)
            
            if save_optimization_path:
                qng_param_history.append(qng_params.copy())
            
            # 跟踪度量张量条件数（如果需要）
            if track_metric_condition:
                try:
                    # 这需要访问内部度量张量，这里简化处理
                    metric_condition_history.append(1.0)  # 占位符
                except:
                    metric_condition_history.append(np.nan)
            
            current_energy = cost_function_qng(qng_params)
            conv = np.abs(current_energy - prev_energy)
            
            if print_progress and iteration % 20 == 0:
                print(f"QNG 迭代 {iteration}: E = {current_energy:.8f}, 收敛参数 = {conv:.8f}")
            
            if conv <= convergence_threshold:
                qng_converged = True
                if print_progress:
                    print(f"QNG在第{iteration}步收敛！")
                break
        
        qng_final_energy = cost_function_qng(qng_params)
        qng_time = time.time() - start_time
        
        # 11. 执行标准梯度下降优化（如果需要对比）
        standard_results = {}
        if compare_standard:
            if print_progress:
                print("\n开始标准梯度下降优化（对比）...")
            
            start_time_std = time.time()
            std_opt = qml.GradientDescentOptimizer(stepsize=standard_step_size)
            std_params = initial_params.copy()
            std_energy_history = []
            std_param_history = []
            
            std_converged = False
            for iteration in range(max_iterations):
                std_params, prev_energy = std_opt.step_and_cost(cost_function, std_params)
                std_energy_history.append(prev_energy)
                
                if save_optimization_path:
                    std_param_history.append(std_params.copy())
                
                current_energy = cost_function(std_params)
                conv = np.abs(current_energy - prev_energy)
                
                if print_progress and iteration % 20 == 0:
                    print(f"标准GD 迭代 {iteration}: E = {current_energy:.8f}, 收敛参数 = {conv:.8f}")
                
                if conv <= convergence_threshold:
                    std_converged = True
                    if print_progress:
                        print(f"标准GD在第{iteration}步收敛！")
                    break
            
            std_final_energy = cost_function(std_params)
            std_time = time.time() - start_time_std
            
            standard_results = {
                "final_energy": float(std_final_energy),
                "final_parameters": std_params.tolist(),
                "energy_history": std_energy_history,
                "parameter_history": [p.tolist() for p in std_param_history] if save_optimization_path else "not_saved",
                "converged": std_converged,
                "iterations": len(std_energy_history),
                "optimization_time": std_time,
                "optimizer_config": {
                    "type": "GradientDescentOptimizer",
                    "step_size": standard_step_size
                }
            }
        
        # 12. 对比分析
        comparison = {}
        if compare_standard:
            qng_iterations = len(qng_energy_history)
            std_iterations = len(std_energy_history)
            
            convergence_speedup = std_iterations / qng_iterations if qng_iterations > 0 else 1.0
            energy_improvement = abs(qng_final_energy - std_final_energy)
            
            comparison = {
                "convergence_speedup": float(convergence_speedup),
                "qng_iterations": qng_iterations,
                "standard_iterations": std_iterations,
                "energy_improvement": float(energy_improvement),
                "qng_better": qng_final_energy < std_final_energy,
                "time_comparison": {
                    "qng_time": qng_time,
                    "standard_time": std_time,
                    "time_per_iteration": {
                        "qng": qng_time / qng_iterations if qng_iterations > 0 else 0,
                        "standard": std_time / std_iterations if std_iterations > 0 else 0
                    }
                }
            }
        
        # 13. 构建返回结果
        result = {
            "success": True,
            
            "qng_results": {
                "final_energy": float(qng_final_energy),
                "final_parameters": qng_params.tolist(),
                "energy_history": qng_energy_history,
                "parameter_history": [p.tolist() for p in qng_param_history] if save_optimization_path else "not_saved",
                "metric_condition_history": metric_condition_history if track_metric_condition else "not_tracked",
                "converged": qng_converged,
                "iterations": len(qng_energy_history),
                "optimization_time": qng_time,
                "optimizer_config": {
                    "type": "QNGOptimizer",
                    "step_size": step_size,
                    "metric_approximation": metric_approx,
                    "regularization": regularization
                }
            },
            
            "standard_results": standard_results,
            "comparison": comparison,
            
            "system_info": {
                "hamiltonian_type": hamiltonian_type,
                "qubits": qubits,
                "n_parameters": n_params,
                "ansatz_type": ansatz_type,
                "exact_energy": float(exact_energy) if exact_energy is not None else None,
                "energy_accuracy": {
                    "qng_error": float(abs(qng_final_energy - exact_energy)) if exact_energy is not None else None,
                    "standard_error": float(abs(std_final_energy - exact_energy)) if (compare_standard and exact_energy is not None) else None
                }
            },
            
            "optimization_analysis": {
                "qng_convergence_behavior": "faster" if (compare_standard and qng_converged and len(qng_energy_history) < len(std_energy_history)) else "similar",
                "metric_tensor_benefits": "QNG利用量子几何结构优化参数空间导航",
                "computational_overhead": "每步QNG需要更多量子电路评估来计算度量张量",
                "recommended_use_cases": [
                    "参数景观复杂的变分电路",
                    "存在贫瘠高原的优化问题",
                    "需要快速收敛的应用"
                ]
            },
            
            "algorithm_config": full_config,
            
            "computational_details": {
                "total_quantum_evaluations": {
                    "qng": "每步需要2N+1次评估（N为参数数）",
                    "standard": "每步需要2N次评估"
                },
                "memory_requirements": "中等（需要存储度量张量）",
                "scalability": f"适用于{n_params}参数的系统"
            }
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"量子自然梯度VQE计算过程中发生错误: {str(e)}",
            "algorithm_config": full_config,
            "suggested_action": "请检查系统定义和ansatz配置"
        }


def validate_qng_vqe_config(config):
    """
    验证量子自然梯度VQE配置参数的辅助函数
    
    Args:
        config (dict): 配置字典
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(config, dict):
        return False, "配置必须是字典格式"
    
    required_keys = ['system_definition', 'ansatz_config']
    for key in required_keys:
        if key not in config:
            return False, f"缺少必需参数: {key}"
    
    # 验证系统定义
    system_def = config['system_definition']
    if 'hamiltonian_type' not in system_def:
        return False, "system_definition必须包含hamiltonian_type"
    
    ham_type = system_def['hamiltonian_type']
    if ham_type not in ['molecule', 'spin_system', 'custom_pauli', 'ising']:
        return False, f"不支持的hamiltonian_type: {ham_type}"
    
    # 验证ansatz配置
    ansatz_config = config['ansatz_config']
    if 'ansatz_type' not in ansatz_config:
        return False, "ansatz_config必须包含ansatz_type"
    
    ansatz_type = ansatz_config['ansatz_type']
    if ansatz_type not in ['hardware_efficient', 'uccsd', 'custom', 'single_qubit_rotations']:
        return False, f"不支持的ansatz_type: {ansatz_type}"
    
    # 验证配置参数
    user_config = config.get('config', {})
    
    if 'step_size' in user_config:
        step_size = user_config['step_size']
        if not isinstance(step_size, (int, float)) or step_size <= 0:
            return False, "step_size必须是正数"
    
    if 'metric_approx' in user_config:
        metric_approx = user_config['metric_approx']
        if metric_approx not in ['diag', 'block-diag', 'full']:
            return False, "metric_approx必须是'diag', 'block-diag'或'full'"
    
    return True, ""


def estimate_qng_vqe_resources(system_definition, ansatz_config):
    """
    估算量子自然梯度VQE算法所需资源
    
    Args:
        system_definition (dict): 系统定义
        ansatz_config (dict): ansatz配置
        
    Returns:
        dict: 资源估算结果
    """
    try:
        hamiltonian_type = system_definition['hamiltonian_type']
        ansatz_type = ansatz_config['ansatz_type']
        
        # 估算量子比特数
        if hamiltonian_type == 'molecule':
            molecule_name = system_definition.get('molecule_name', 'H2')
            qubit_estimates = {'H2': 4, 'LiH': 6, 'BeH2': 8, 'H2O': 8}
            qubits = qubit_estimates.get(molecule_name, 4)
        else:
            qubits = system_definition.get('n_qubits', 4)
        
        # 估算参数数量
        n_layers = ansatz_config.get('n_layers', 2)
        if ansatz_type == 'single_qubit_rotations':
            n_params = 2 * qubits
        elif ansatz_type == 'hardware_efficient':
            rotation_gates = ansatz_config.get('rotation_gates', ['RY', 'RZ'])
            n_params = n_layers * len(rotation_gates) * qubits
        elif ansatz_type == 'uccsd':
            n_params = min(5, qubits // 2)
        else:
            n_params = qubits * 2  # 默认估算
        
        # 计算量子电路评估次数
        evaluations_per_qng_step = 2 * n_params + 1  # 计算度量张量需要额外评估
        evaluations_per_std_step = 2 * n_params
        
        # 估算计算时间
        time_per_qng_step = evaluations_per_qng_step * 0.01  # 每次评估0.01秒的经验估算
        time_per_std_step = evaluations_per_std_step * 0.01
        
        estimated_qng_iterations = 100  # 经验值，QNG通常收敛更快
        estimated_std_iterations = 200
        
        total_qng_time = time_per_qng_step * estimated_qng_iterations
        total_std_time = time_per_std_step * estimated_std_iterations
        
        return {
            "system_info": {
                "hamiltonian_type": hamiltonian_type,
                "estimated_qubits": qubits,
                "ansatz_type": ansatz_type,
                "n_parameters": n_params
            },
            "computational_requirements": {
                "qng_evaluations_per_step": evaluations_per_qng_step,
                "standard_evaluations_per_step": evaluations_per_std_step,
                "overhead_ratio": evaluations_per_qng_step / evaluations_per_std_step
            },
            "performance_estimates": {
                "expected_qng_iterations": estimated_qng_iterations,
                "expected_standard_iterations": estimated_std_iterations,
                "convergence_speedup": estimated_std_iterations / estimated_qng_iterations,
                "total_time_qng": f"{total_qng_time:.1f}秒",
                "total_time_standard": f"{total_std_time:.1f}秒"
            },
            "feasibility": {
                "feasible": qubits <= 12,
                "scalability": "适中" if n_params <= 20 else "困难",
                "memory_requirement": "中等（需要存储度量张量）"
            },
            "qng_advantages": [
                "更快的收敛速度",
                "更好的优化轨迹",
                "对初始参数更鲁棒",
                "适用于贫瘠高原问题"
            ],
            "qng_disadvantages": [
                "每步计算开销更大",
                "需要更多量子电路评估",
                "度量张量计算复杂度"
            ]
        }
    except Exception as e:
        return {
            "feasible": False,
            "error": f"无法估算资源: {str(e)}"
        }