# meta_vqe_template.py
import pennylane as qml
from pennylane import numpy as np
import time
from functools import partial

def run_meta_vqe(config):
    """
    Meta-VQE算法参数化模板
    
    Args:
        config (dict): 包含system_config, training_config和可选配置参数的字典
            - system_config (dict): 系统配置
            - training_config (dict): 训练配置
            - ansatz_config (dict, optional): ansatz配置
            - evaluation_config (dict, optional): 评估配置
            - print_progress (bool, optional): 打印进度，默认True
            - save_training_history (bool, optional): 保存训练历史，默认True
            - plot_results (bool, optional): 生成图表，默认False
    
    Returns:
        dict: 包含计算结果和完整配置信息的字典
    """
    
    # 1. 提取和验证输入参数
    system_config = config.get('system_config')
    training_config = config.get('training_config')
    
    if not system_config or not training_config:
        return {
            "success": False,
            "error": "缺少必需参数: system_config 和 training_config",
            "suggested_action": "请提供系统配置和训练配置"
        }
    
    # 2. 设置默认参数
    ansatz_config = config.get('ansatz_config', {})
    evaluation_config = config.get('evaluation_config', {})
    user_config = config.get('config', {})
    
    # 系统参数
    hamiltonian_type = system_config['hamiltonian_type']
    n_qubits = system_config.get('n_qubits', 4)
    parameter_name = system_config['parameter_name']
    fixed_params = system_config.get('fixed_params', {'eta': 0.75})
    boundary_conditions = system_config.get('boundary_conditions', 'periodic')
    
    # 训练参数
    training_points = training_config.get('training_points')
    if training_points is None:
        training_range = training_config.get('training_range', {})
        min_val = training_range.get('min_value', -1.0)
        max_val = training_range.get('max_value', 1.0)
        n_train = training_range.get('n_points', 5)
        training_points = np.linspace(min_val, max_val, n_train).tolist()
    
    training_epochs = training_config.get('training_epochs', 100)
    learning_rate = training_config.get('learning_rate', 0.01)
    optimizer_type = training_config.get('optimizer', 'adam')
    
    # Ansatz参数
    encoding_type = ansatz_config.get('encoding_type', 'linear')
    n_encoding_layers = ansatz_config.get('n_encoding_layers', 1)
    n_processing_layers = ansatz_config.get('n_processing_layers', 3)
    entangling_gates = ansatz_config.get('entangling_gates', 'CNOT')
    parameter_init = ansatz_config.get('parameter_init', 'random')
    
    # 评估参数
    test_points = evaluation_config.get('test_points')
    if test_points is None:
        test_range = evaluation_config.get('test_range', {})
        test_min = test_range.get('min_value', -1.2)
        test_max = test_range.get('max_value', 1.2)
        n_test = test_range.get('n_points', 50)
        test_points = np.linspace(test_min, test_max, n_test).tolist()
    
    compute_exact = evaluation_config.get('compute_exact', True)
    interpolation_test = evaluation_config.get('interpolation_test', True)
    extrapolation_test = evaluation_config.get('extrapolation_test', False)
    
    # 其他参数
    print_progress = user_config.get('print_progress', True)
    save_training_history = user_config.get('save_training_history', True)
    plot_results = user_config.get('plot_results', False)
    
    # 3. 构建完整配置记录
    full_config = {
        "algorithm_name": "Meta-VQE",
        "algorithm_type": "meta_variational_quantum_eigensolver",
        "parameters_used": {
            "system_config": system_config,
            "training_config": training_config,
            "ansatz_config": ansatz_config,
            "evaluation_config": evaluation_config,
            "user_config": user_config
        },
        "training_points": training_points,
        "test_points": test_points,
        "execution_environment": {
            "device": "default.qubit",
            "interface": "autograd"
        }
    }
    
    try:
        # 4. 检查量子比特限制
        if n_qubits > 10:
            return {
                "success": False,
                "error": f"量子比特数{n_qubits}超过Meta-VQE限制(10个)",
                "algorithm_config": full_config,
                "suggested_action": "请减少量子比特数"
            }
        
        # 5. 构建参数化Hamiltonian生成函数
        def build_hamiltonian(param_value):
            """根据参数值构建Hamiltonian"""
            if hamiltonian_type == 'xxz_spin_chain':
                return build_xxz_hamiltonian(n_qubits, param_value, 
                                           fixed_params.get('eta', 0.75),
                                           boundary_conditions)
            elif hamiltonian_type == 'ising_model':
                return build_ising_hamiltonian(n_qubits, param_value,
                                             fixed_params.get('h', 1.0))
            else:
                raise ValueError(f"不支持的Hamiltonian类型: {hamiltonian_type}")
        
        # 6. 创建量子设备
        dev = qml.device("default.qubit", wires=n_qubits)
        
        # 7. 构建Meta-VQE ansatz
        def build_meta_ansatz(encoding_type, n_encoding_layers, n_processing_layers):
            """构建Meta-VQE变分电路"""
            
            # 计算参数维度
            if encoding_type == 'linear':
                encoding_params_per_qubit = 4  # RZ(θ₀*λ + θ₁), RY(θ₂*λ + θ₃)
            else:
                encoding_params_per_qubit = 4  # 简化，都用4个参数
            
            processing_params_per_qubit = 2  # RZ(θ), RY(θ)
            
            total_encoding_params = n_encoding_layers * n_qubits * encoding_params_per_qubit
            total_processing_params = n_processing_layers * n_qubits * processing_params_per_qubit
            
            param_shape = (n_encoding_layers + n_processing_layers, n_qubits, 4)
            
            def meta_ansatz(params, parameter_value):
                """Meta-VQE变分电路"""
                layer_idx = 0
                
                # 编码层
                for enc_layer in range(n_encoding_layers):
                    for qubit in range(n_qubits):
                        if encoding_type == 'linear':
                            # 线性编码: RZ(θ₀*λ + θ₁), RY(θ₂*λ + θ₃)
                            qml.RZ(params[layer_idx][qubit][0] * parameter_value + 
                                  params[layer_idx][qubit][1], wires=qubit)
                            qml.RY(params[layer_idx][qubit][2] * parameter_value + 
                                  params[layer_idx][qubit][3], wires=qubit)
                        elif encoding_type == 'trigonometric':
                            # 三角编码
                            qml.RZ(params[layer_idx][qubit][0] * np.sin(parameter_value) + 
                                  params[layer_idx][qubit][1], wires=qubit)
                            qml.RY(params[layer_idx][qubit][2] * np.cos(parameter_value) + 
                                  params[layer_idx][qubit][3], wires=qubit)
                    
                    # 纠缠层
                    if entangling_gates == 'CNOT':
                        for i in range(0, n_qubits - 1, 2):
                            qml.CNOT(wires=[i, i + 1])
                        if n_qubits > 2:
                            qml.CNOT(wires=[n_qubits - 1, 0])  # 周期边界条件
                    
                    layer_idx += 1
                
                # 处理层
                for proc_layer in range(n_processing_layers):
                    for qubit in range(n_qubits):
                        qml.RZ(params[layer_idx][qubit][0], wires=qubit)
                        qml.RY(params[layer_idx][qubit][2], wires=qubit)
                    
                    # 纠缠层
                    if entangling_gates == 'CNOT':
                        for i in range(0, n_qubits - 1, 2):
                            qml.CNOT(wires=[i, i + 1])
                    
                    layer_idx += 1
            
            return meta_ansatz, param_shape
        
        meta_ansatz, param_shape = build_meta_ansatz(encoding_type, n_encoding_layers, n_processing_layers)
        
        # 8. 初始化参数
        if parameter_init == 'random':
            params = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=param_shape)
        elif parameter_init == 'zero':
            params = np.zeros(param_shape)
        else:  # small_random
            params = np.random.uniform(low=-0.1, high=0.1, size=param_shape)
        
        # 9. 定义期望值计算函数
        def compute_expectation_value(hamiltonian):
            """为给定Hamiltonian创建期望值计算函数"""
            coeffs, observables = hamiltonian.terms()
            qnodes = qml.map(meta_ansatz, observables, dev)
            cost = qml.dot(coeffs, qnodes)
            return cost
        
        # 10. 定义Meta-VQE成本函数
        def meta_vqe_cost(params):
            """Meta-VQE成本函数：对所有训练点求和"""
            total_cost = 0.0
            for param_val in training_points:
                hamiltonian = build_hamiltonian(param_val)
                expectation_fn = compute_expectation_value(hamiltonian)
                total_cost += expectation_fn(params, parameter_value=param_val)
            return total_cost
        
        # 11. 设置优化器
        if optimizer_type == 'adam':
            optimizer = qml.AdamOptimizer(stepsize=learning_rate)
        elif optimizer_type == 'adagrad':
            optimizer = qml.AdagradOptimizer(stepsize=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = qml.GradientDescentOptimizer(stepsize=learning_rate)
        else:
            optimizer = qml.AdamOptimizer(stepsize=learning_rate)
        
        # 12. 执行训练
        start_time = time.time()
        
        training_costs = []
        training_params_history = []
        
        if print_progress:
            print("开始Meta-VQE训练...")
            print(f"系统: {hamiltonian_type}, 量子比特: {n_qubits}")
            print(f"编码参数: {parameter_name}, 训练点: {len(training_points)}")
            print(f"Ansatz: {n_encoding_layers}编码层 + {n_processing_layers}处理层")
        
        for epoch in range(training_epochs):
            params, cost = optimizer.step_and_cost(meta_vqe_cost, params)
            
            training_costs.append(float(cost))
            if save_training_history:
                training_params_history.append(params.copy())
            
            if print_progress and epoch % 20 == 0:
                print(f"Epoch {epoch}: 成本函数 = {cost:.6f}")
        
        training_time = time.time() - start_time
        
        if print_progress:
            print(f"训练完成！最终成本: {training_costs[-1]:.6f}")
        
        # 13. 在测试点上评估性能
        meta_predictions = []
        exact_energies = []
        
        if print_progress:
            print("在测试点上评估Meta-VQE性能...")
        
        for param_val in test_points:
            # Meta-VQE预测
            hamiltonian = build_hamiltonian(param_val)
            expectation_fn = compute_expectation_value(hamiltonian)
            predicted_energy = expectation_fn(params, parameter_value=param_val)
            meta_predictions.append(float(predicted_energy))
            
            # 精确解（如果需要）
            if compute_exact:
                exact_energy = compute_exact_ground_state(hamiltonian)
                exact_energies.append(float(exact_energy))
        
        # 14. 计算评估指标
        evaluation_metrics = {}
        
        if compute_exact:
            # 计算预测误差
            errors = np.array(meta_predictions) - np.array(exact_energies)
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))
            max_error = np.max(np.abs(errors))
            
            evaluation_metrics = {
                "mean_absolute_error": float(mae),
                "root_mean_square_error": float(rmse),
                "max_absolute_error": float(max_error),
                "prediction_accuracy": float(1.0 - mae / np.mean(np.abs(exact_energies)))
            }
            
            # 插值和外推测试
            if interpolation_test:
                # 测试在训练点之间的插值能力
                train_min, train_max = min(training_points), max(training_points)
                interp_points = [p for p in test_points if train_min <= p <= train_max]
                if interp_points:
                    interp_errors = [abs(meta_predictions[i] - exact_energies[i]) 
                                   for i, p in enumerate(test_points) if p in interp_points]
                    evaluation_metrics["interpolation_mae"] = float(np.mean(interp_errors))
            
            if extrapolation_test:
                # 测试外推能力
                train_min, train_max = min(training_points), max(training_points)
                extrap_points = [p for p in test_points if p < train_min or p > train_max]
                if extrap_points:
                    extrap_errors = [abs(meta_predictions[i] - exact_energies[i])
                                   for i, p in enumerate(test_points) if p in extrap_points]
                    evaluation_metrics["extrapolation_mae"] = float(np.mean(extrap_errors))
        
        # 15. 构建返回结果
        result = {
            "success": True,
            
            "training_results": {
                "final_cost": training_costs[-1],
                "cost_history": training_costs,
                "parameter_history": [p.tolist() for p in training_params_history] if save_training_history else "not_saved",
                "training_time": training_time,
                "epochs_completed": training_epochs,
                "optimizer_used": optimizer_type,
                "converged": len(training_costs) > 10 and 
                           abs(training_costs[-1] - training_costs[-10]) < 1e-6
            },
            
            "predictions": {
                "test_parameters": test_points,
                "meta_vqe_predictions": meta_predictions,
                "exact_ground_states": exact_energies if compute_exact else "not_computed"
            },
            
            "evaluation_metrics": evaluation_metrics,
            
            "learned_ansatz": {
                "final_parameters": params.tolist(),
                "parameter_shape": param_shape,
                "encoding_type": encoding_type,
                "n_encoding_layers": n_encoding_layers,
                "n_processing_layers": n_processing_layers,
                "total_parameters": int(np.prod(param_shape))
            },
            
            "comparison_with_exact": {
                "exact_computation_enabled": compute_exact,
                "prediction_quality": "good" if (compute_exact and evaluation_metrics.get("prediction_accuracy", 0) > 0.95) else "needs_improvement",
                "best_predicted_point": test_points[np.argmin(np.abs(meta_predictions))] if meta_predictions else None
            },
            
            "algorithm_config": full_config,
            
            "computational_details": {
                "total_training_evaluations": training_epochs * len(training_points),
                "total_test_evaluations": len(test_points),
                "quantum_advantage": f"一次训练预测{len(test_points)}个点，相比{len(test_points)}次VQE节省大量计算",
                "parameter_encoding_method": encoding_type,
                "learned_parameter_range": f"{min(training_points):.2f} to {max(training_points):.2f}",
                "test_parameter_range": f"{min(test_points):.2f} to {max(test_points):.2f}"
            }
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Meta-VQE计算过程中发生错误: {str(e)}",
            "algorithm_config": full_config,
            "suggested_action": "请检查系统配置和训练参数设置"
        }


def build_xxz_hamiltonian(n_qubits, delta, eta, boundary_conditions='periodic'):
    """构建XXZ自旋链Hamiltonian"""
    coeffs = []
    ops = []
    
    # 相互作用项
    for i in range(n_qubits - 1):
        # XX + YY 项
        ops.extend([qml.PauliX(i) @ qml.PauliX(i+1), qml.PauliY(i) @ qml.PauliY(i+1)])
        coeffs.extend([1.0, 1.0])
        
        # Delta * ZZ 项
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i+1))
        coeffs.append(delta)
    
    # 周期边界条件
    if boundary_conditions == 'periodic' and n_qubits > 2:
        ops.extend([qml.PauliX(n_qubits-1) @ qml.PauliX(0), 
                   qml.PauliY(n_qubits-1) @ qml.PauliY(0)])
        coeffs.extend([1.0, 1.0])
        ops.append(qml.PauliZ(n_qubits-1) @ qml.PauliZ(0))
        coeffs.append(delta)
    
    # 横向场项
    for i in range(n_qubits):
        ops.append(qml.PauliZ(i))
        coeffs.append(eta)
    
    return qml.Hamiltonian(coeffs, ops, simplify=True)


def build_ising_hamiltonian(n_qubits, coupling, field):
    """构建横场Ising模型Hamiltonian"""
    coeffs = []
    ops = []
    
    # ZZ相互作用
    for i in range(n_qubits - 1):
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i+1))
        coeffs.append(-coupling)  # 负号表示铁磁耦合
    
    # 横向磁场
    for i in range(n_qubits):
        ops.append(qml.PauliX(i))
        coeffs.append(-field)
    
    return qml.Hamiltonian(coeffs, ops, simplify=True)


def compute_exact_ground_state(hamiltonian):
    """计算Hamiltonian的精确基态能量"""
    matrix = qml.matrix(hamiltonian)
    eigenvalues = np.linalg.eigvals(matrix)
    return np.real(np.min(eigenvalues))


def validate_meta_vqe_config(config):
    """
    验证Meta-VQE配置参数的辅助函数
    
    Args:
        config (dict): 配置字典
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(config, dict):
        return False, "配置必须是字典格式"
    
    required_keys = ['system_config', 'training_config']
    for key in required_keys:
        if key not in config:
            return False, f"缺少必需参数: {key}"
    
    # 验证系统配置
    system_config = config['system_config']
    if 'hamiltonian_type' not in system_config:
        return False, "system_config必须包含hamiltonian_type"
    
    ham_type = system_config['hamiltonian_type']
    if ham_type not in ['xxz_spin_chain', 'ising_model', 'heisenberg_model', 'molecular_system', 'custom']:
        return False, f"不支持的hamiltonian_type: {ham_type}"
    
    if 'parameter_name' not in system_config:
        return False, "system_config必须包含parameter_name"
    
    # 验证训练配置
    training_config = config['training_config']
    training_points = training_config.get('training_points')
    training_range = training_config.get('training_range')
    
    if training_points is None and training_range is None:
        return False, "必须指定training_points或training_range"
    
    if training_points is not None:
        if not isinstance(training_points, list) or len(training_points) < 2:
            return False, "training_points必须是至少包含2个元素的列表"
    
    return True, ""


def estimate_meta_vqe_resources(system_config, training_config, ansatz_config):
    """
    估算Meta-VQE算法所需资源
    
    Args:
        system_config (dict): 系统配置
        training_config (dict): 训练配置  
        ansatz_config (dict): ansatz配置
        
    Returns:
        dict: 资源估算结果
    """
    try:
        n_qubits = system_config.get('n_qubits', 4)
        hamiltonian_type = system_config['hamiltonian_type']
        
        # 训练点数量
        training_points = training_config.get('training_points')
        if training_points:
            n_training_points = len(training_points)
        else:
            training_range = training_config.get('training_range', {})
            n_training_points = training_range.get('n_points', 5)
        
        training_epochs = training_config.get('training_epochs', 100)
        
        # Ansatz参数
        n_encoding_layers = ansatz_config.get('n_encoding_layers', 1)
        n_processing_layers = ansatz_config.get('n_processing_layers', 3)
        
        # 计算参数数量
        total_layers = n_encoding_layers + n_processing_layers
        total_params = total_layers * n_qubits * 4
        
        # 计算量子电路评估次数
        evaluations_per_epoch = n_training_points * (2 * total_params + 1)  # 梯度计算
        total_training_evaluations = training_epochs * evaluations_per_epoch
        
        # 估算时间
        time_per_evaluation = 0.01  # 每次评估0.01秒的经验估算
        estimated_training_time = total_training_evaluations * time_per_evaluation
        
        # 与多次VQE的对比
        standard_vqe_evaluations_per_point = 100 * (2 * total_params + 1)  # 假设每个VQE需要100次迭代
        
        return {
            "system_info": {
                "hamiltonian_type": hamiltonian_type,
                "n_qubits": n_qubits,
                "n_training_points": n_training_points,
                "ansatz_layers": f"{n_encoding_layers}编码 + {n_processing_layers}处理"
            },
            "parameter_analysis": {
                "total_parameters": total_params,
                "encoding_parameters": n_encoding_layers * n_qubits * 4,
                "processing_parameters": n_processing_layers * n_qubits * 4,
                "parameter_encoding": "每个编码层每量子比特4参数"
            },
            "training_requirements": {
                "total_training_evaluations": total_training_evaluations,
                "evaluations_per_epoch": evaluations_per_epoch,
                "estimated_training_time": f"{estimated_training_time:.1f}秒"
            },
            "efficiency_comparison": {
                "meta_vqe_advantage": f"一次训练可预测任意参数值",
                "vs_multiple_vqe": f"相比{n_training_points}次标准VQE节省大量计算",
                "prediction_cost": "预测新参数点几乎无额外成本",
                "scalability": "训练成本与参数点数线性相关"
            },
            "memory_requirements": {
                "parameter_storage": f"{total_params}个浮点数",
                "training_history": "可选择保存训练历史",
                "quantum_state": f"2^{n_qubits}复数振幅（模拟）"
            },
            "feasibility": {
                "feasible": n_qubits <= 10 and total_params <= 200,
                "quantum_advantage": "参数插值和外推能力",
                "recommended_use": "参数扫描、相变研究、势能面分析"
            }
        }
    except Exception as e:
        return {
            "feasible": False,
            "error": f"无法估算资源: {str(e)}"
        }