# spin_sector_vqe_template.py
import pennylane as qml
from pennylane import numpy as np
import jax
from jax import numpy as jnp, random
import optax
import time

# 尝试导入Catalyst
try:
    import catalyst
    CATALYST_AVAILABLE = True
except ImportError:
    CATALYST_AVAILABLE = False

def run_spin_sector_vqe(config):
    """
    自旋扇区VQE算法参数化模板
    
    Args:
        config (dict): 包含symbols, coordinates和可选配置参数的字典
            - symbols (list): 原子符号列表
            - coordinates (list): 原子坐标数组
            - electrons (int, optional): 电子数
            - target_spin_sector (int, optional): 目标自旋扇区，默认0
            - delta_sz (int, optional): 自旋投影变化
            - max_iterations (int, optional): 最大迭代次数，默认100
            - learning_rate (float, optional): 学习率，默认0.8
            - optimizer (str, optional): 优化器类型，默认'sgd'
            - convergence_threshold (float, optional): 收敛阈值，默认1e-6
            - calculate_total_spin (bool, optional): 是否计算总自旋，默认True
            - initial_state (str, optional): 初始态选择，默认'hf'
    
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
    electrons = user_config.get('electrons')
    target_spin_sector = user_config.get('target_spin_sector', 0)
    delta_sz = user_config.get('delta_sz')
    max_iterations = user_config.get('max_iterations', 100)
    learning_rate = user_config.get('learning_rate', 0.8)
    optimizer_type = user_config.get('optimizer', 'sgd')
    convergence_threshold = user_config.get('convergence_threshold', 1e-6)
    calculate_total_spin = user_config.get('calculate_total_spin', True)
    initial_state = user_config.get('initial_state', 'hf')
    
    # 根据目标自旋扇区自动设置delta_sz
    if delta_sz is None:
        if target_spin_sector == 0:
            delta_sz = 0  # 自旋保持激发，用于基态
        elif target_spin_sector == 1:
            delta_sz = 1  # 自旋翻转激发，用于三重态
        else:
            delta_sz = 0
    
    # 3. 构建完整配置记录
    full_config = {
        "algorithm_name": "Spin-Sector VQE",  
        "algorithm_type": "spin_resolved_variational_quantum_eigensolver",
        "parameters_used": {
            "symbols": symbols,
            "coordinates": coordinates,
            "electrons": electrons,
            "target_spin_sector": target_spin_sector,
            "delta_sz": delta_sz,
            "max_iterations": max_iterations,
            "learning_rate": learning_rate,
            "optimizer": optimizer_type,
            "convergence_threshold": convergence_threshold,
            "calculate_total_spin": calculate_total_spin,
            "initial_state": initial_state
        },
        "default_values_applied": {
            "target_spin_sector": target_spin_sector == 0,
            "max_iterations": max_iterations == 100,
            "learning_rate": learning_rate == 0.8,
            "optimizer": optimizer_type == 'sgd',
            "convergence_threshold": convergence_threshold == 1e-6,
            "calculate_total_spin": calculate_total_spin == True,
            "initial_state": initial_state == 'hf',
            "electrons": electrons is None,
            "delta_sz": user_config.get('delta_sz') is None
        },
        "execution_environment": {
            "device": "lightning.qubit",
            "interface": "jax",
            "catalyst_available": CATALYST_AVAILABLE
        }
    }
    
    try:
        # 4. 构建分子Hamiltonian
        try:
            coordinates_array = np.array(coordinates)
            H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates_array)
        except Exception as e:
            return {
                "success": False,
                "error": f"无法构建分子Hamiltonian: {str(e)}",
                "algorithm_config": full_config,
                "suggested_action": "请检查分子几何结构和参数设置"
            }
        
        # 5. 检查量子比特限制
        if qubits > 12:
            return {
                "success": False,
                "error": f"分子需要{qubits}量子比特，超过自旋扇区VQE限制(12个)",
                "algorithm_config": full_config,
                "estimated_qubits": qubits,
                "suggested_action": "请选择更小的分子"
            }
        
        # 6. 确定电子数
        if electrons is None:
            electrons = sum([_get_atomic_number(symbol) for symbol in symbols])
            full_config["parameters_used"]["electrons"] = electrons
        
        # 7. 验证自旋扇区兼容性
        if target_spin_sector == 1 and electrons % 2 == 0:
            # 偶数电子的三重态需要特殊处理
            pass
        elif target_spin_sector == 0 and electrons % 2 == 1:
            return {
                "success": False,
                "error": f"奇数电子({electrons})的分子无法形成S=0基态",
                "algorithm_config": full_config,
                "suggested_action": "请选择target_spin_sector=1或检查电子数"
            }
        
        # 8. 生成Hartree-Fock态和激发
        hf_state = qml.qchem.hf_state(electrons, qubits)
        singles, doubles = qml.qchem.excitations(electrons, qubits, delta_sz=delta_sz)
        
        # 9. 构建总自旋算符(如果需要)
        if calculate_total_spin:
            S2 = qml.qchem.spin2(electrons, qubits)
        
        # 10. 创建量子设备
        dev = qml.device("lightning.qubit", wires=qubits)
        
        # 11. 定义变分电路
        def circuit(params, wires):
            # 选择初始态
            if initial_state == 'hf':
                initial = hf_state
            elif initial_state == 'flipped_hf':
                initial = np.flip(hf_state)  # 用于激发态
            else:
                initial = hf_state
            
            qml.AllSinglesDoubles(params, wires, initial, singles, doubles)
        
        # 12. 定义成本函数
        @qml.qnode(dev, interface="jax")
        def cost_fn(params):
            circuit(params, wires=range(qubits))
            return qml.expval(H)
        
        # 13. 定义总自旋期望值函数(如果需要)
        if calculate_total_spin:
            @qml.qnode(dev, interface="jax")
            def S2_exp_value(params):
                circuit(params, wires=range(qubits))
                return qml.expval(S2)
            
            def total_spin(params):
                """从S²期望值计算总自旋S"""
                s2_val = S2_exp_value(params)
                return -0.5 + jnp.sqrt(0.25 + s2_val)
        
        # 14. 设置优化器
        if optimizer_type == 'sgd':
            opt = optax.sgd(learning_rate=learning_rate)
        elif optimizer_type == 'adam':
            opt = optax.adam(learning_rate=learning_rate)
        elif optimizer_type == 'adamw':
            opt = optax.adamw(learning_rate=learning_rate)
        elif optimizer_type == 'rmsprop':
            opt = optax.rmsprop(learning_rate=learning_rate)
        else:
            opt = optax.sgd(learning_rate=learning_rate)
            full_config["parameters_used"]["optimizer"] = 'sgd'
        
        # 15. 初始化参数
        n_params = len(singles) + len(doubles)
        if n_params == 0:
            return {
                "success": False,
                "error": f"没有找到可用的激发(delta_sz={delta_sz})",
                "algorithm_config": full_config,
                "suggested_action": "请尝试不同的delta_sz值或目标自旋扇区"
            }
        
        key = random.PRNGKey(0)
        init_params = random.normal(key, shape=(n_params,)) * np.pi
        
        # 16. 执行优化
        start_time = time.time()
        
        if CATALYST_AVAILABLE:
            # 使用Catalyst JIT编译优化
            @qml.qjit
            def update_step(i, params, opt_state):
                grads = qml.grad(cost_fn)(params)
                updates, opt_state = opt.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return params, opt_state
            
            opt_state = opt.init(init_params)
            params = init_params
            energy_history = []
            
            for i in range(max_iterations):
                params, opt_state = update_step(i, params, opt_state)
                energy = cost_fn(params)
                energy_history.append(float(energy))
                
                # 检查收敛
                if i > 0 and abs(energy_history[-1] - energy_history[-2]) < convergence_threshold:
                    break
        else:
            # 标准JAX优化
            opt_state = opt.init(init_params)
            params = init_params
            energy_history = []
            
            for i in range(max_iterations):
                energy = cost_fn(params)
                grads = jax.grad(cost_fn)(params)
                
                energy_history.append(float(energy))
                
                updates, opt_state = opt.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                
                # 检查收敛
                if i > 0 and abs(energy_history[-1] - energy_history[-2]) < convergence_threshold:
                    break
        
        end_time = time.time()
        
        # 17. 计算最终结果
        final_energy = float(cost_fn(params))
        
        # 计算总自旋信息
        spin_info = {}
        if calculate_total_spin:
            s2_value = float(S2_exp_value(params))
            s_value = float(total_spin(params))
            spin_info = {
                "S2_expectation_value": s2_value,
                "total_spin_S": s_value,
                "spin_multiplicity": int(2 * s_value + 1),
                "target_spin_sector": target_spin_sector,
                "delta_sz_used": delta_sz
            }
        
        # 18. 构建返回结果
        result = {
            "success": True,
            "molecule": {
                "symbols": symbols,
                "coordinates": coordinates,
                "electrons": electrons
            },
            "energy": final_energy,
            "energy_unit": "Hartree",
            
            "spin_info": spin_info,
            
            "algorithm_config": full_config,
            
            "optimization_info": {
                "converged": len(energy_history) < max_iterations or (
                    len(energy_history) >= 2 and 
                    abs(energy_history[-1] - energy_history[-2]) < convergence_threshold
                ),
                "iterations_performed": len(energy_history),
                "max_iterations_allowed": max_iterations,
                "final_parameters": params.tolist(),
                "initial_energy": energy_history[0] if energy_history else final_energy,
                "final_energy": final_energy,
                "energy_improvement": energy_history[0] - final_energy if energy_history else 0.0,
                "convergence_threshold": convergence_threshold
            },
            
            "system_info": {
                "qubits_used": qubits,
                "electrons": electrons,
                "n_parameters": n_params,
                "single_excitations": len(singles),
                "double_excitations": len(doubles),
                "excitation_details": {
                    "singles": singles,
                    "doubles": doubles
                },
                "hartree_fock_state": hf_state.tolist(),
                "initial_state_used": initial_state
            },
            
            "computational_details": {
                "device_type": "lightning.qubit",
                "backend": "JAX",
                "catalyst_used": CATALYST_AVAILABLE,
                "optimizer_used": optimizer_type,
                "learning_rate_used": learning_rate,
                "execution_time": end_time - start_time
            }
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"自旋扇区VQE计算过程中发生错误: {str(e)}",
            "molecule": {"symbols": symbols, "coordinates": coordinates},
            "algorithm_config": full_config,
            "suggested_action": "请检查输入参数或联系技术支持"
        }


def _get_atomic_number(symbol):
    """获取原子序数"""
    atomic_numbers = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10
    }
    return atomic_numbers.get(symbol, 1)


def validate_spin_sector_vqe_config(config):
    """
    验证自旋扇区VQE配置参数的辅助函数
    
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
    
    # 验证自旋扇区参数
    if 'target_spin_sector' in user_config:
        tss = user_config['target_spin_sector']
        if tss not in [0, 1]:
            return False, "target_spin_sector必须是0或1"
    
    if 'delta_sz' in user_config:
        dsz = user_config['delta_sz']
        if dsz not in [-1, 0, 1]:
            return False, "delta_sz必须是-1, 0或1"
    
    return True, ""


def estimate_spin_sector_vqe_resources(symbols, coordinates, target_spin_sector=0):
    """
    估算自旋扇区VQE算法所需资源
    
    Args:
        symbols (list): 原子符号列表
        coordinates (list): 原子坐标
        target_spin_sector (int): 目标自旋扇区
        
    Returns:
        dict: 资源估算结果
    """
    try:
        coordinates_array = np.array(coordinates)
        H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates_array)
        
        electrons = sum([_get_atomic_number(symbol) for symbol in symbols])
        
        # 估算不同自旋扇区的激发数量
        delta_sz = 0 if target_spin_sector == 0 else 1
        singles, doubles = qml.qchem.excitations(electrons, qubits, delta_sz=delta_sz)
        
        return {
            "qubits_required": qubits,
            "electrons": electrons,
            "target_spin_sector": target_spin_sector,
            "excitations_available": {
                "singles": len(singles),
                "doubles": len(doubles),
                "total": len(singles) + len(doubles)
            },
            "feasible": qubits <= 12,
            "spin_compatibility": _check_spin_compatibility(electrons, target_spin_sector),
            "estimated_time": f"{max_iterations * 0.1:.1f}秒",
            "computational_complexity": "中等"
        }
    except Exception as e:
        return {
            "qubits_required": "unknown",
            "feasible": False,
            "error": f"无法分析分子: {str(e)}"
        }


def _check_spin_compatibility(electrons, target_spin_sector):
    """检查电子数与自旋扇区的兼容性"""
    if target_spin_sector == 0:
        return "兼容" if electrons % 2 == 0 else "不兼容(奇数电子无法形成S=0态)"
    elif target_spin_sector == 1:
        return "兼容"
    else:
        return "未知"