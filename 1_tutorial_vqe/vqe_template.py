# vqe_template.py
import jax
from jax import numpy as np
import pennylane as qml
import optax

# JAX配置
jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)

def run_vqe(config):
    """
    VQE算法参数化模板
    
    Args:
        config (dict): 包含molecule和可选配置参数的字典
            - molecule (str): 分子标识符
            - electrons (int, optional): 电子数
            - max_iterations (int, optional): 最大迭代次数，默认100
            - conv_tol (float, optional): 收敛容忍度，默认1e-6
            - learning_rate (float, optional): 学习率，默认0.4
            - optimizer (str, optional): 优化器类型，默认'sgd'
    
    Returns:
        dict: 包含计算结果和完整配置信息的字典
    """
    
    # 1. 提取和验证输入参数
    molecule = config.get('molecule')
    if not molecule:
        return {
            "success": False,
            "error": "缺少必需参数: molecule",
            "suggested_action": "请提供分子标识符，如H2, LiH等"
        }
    
    # 2. 设置默认参数并记录
    user_config = config.get('config', {})
    electrons = user_config.get('electrons')
    max_iterations = user_config.get('max_iterations', 100)
    conv_tol = user_config.get('conv_tol', 1e-6)
    learning_rate = user_config.get('learning_rate', 0.4)
    optimizer_type = user_config.get('optimizer', 'sgd')
    
    # 3. 构建完整配置记录
    full_config = {
        "algorithm_name": "VQE",
        "algorithm_type": "variational_quantum_eigensolver",
        "parameters_used": {
            "molecule": molecule,
            "electrons": electrons,
            "max_iterations": max_iterations,
            "conv_tol": conv_tol,
            "learning_rate": learning_rate,
            "optimizer": optimizer_type
        },
        "default_values_applied": {
            "max_iterations": max_iterations == 100,
            "conv_tol": conv_tol == 1e-6,
            "learning_rate": learning_rate == 0.4,
            "optimizer": optimizer_type == 'sgd',
            "electrons": electrons is None
        },
        "execution_environment": {
            "device": "lightning.qubit",
            "interface": "jax",
            "basis_set": "minimal",
            "ansatz": "DoubleExcitation",
            "mapping": "Jordan-Wigner"
        }
    }
    
    try:
        # 4. 加载分子Hamiltonian数据
        try:
            dataset = qml.data.load('qchem', molname=molecule)[0]
            H = dataset.hamiltonian
            qubits = len(dataset.hamiltonian.wires)
        except Exception as e:
            return {
                "success": False,
                "error": f"无法加载分子数据: {molecule}. 错误: {str(e)}",
                "algorithm_config": full_config,
                "suggested_action": "请检查分子名称是否正确，支持的分子包括: H2, LiH, BeH2, H2O等"
            }
        
        # 5. 检查量子比特限制
        if qubits > 12:
            return {
                "success": False,
                "error": f"分子{molecule}需要{qubits}量子比特，超过系统限制(12个)",
                "algorithm_config": full_config,
                "estimated_qubits": qubits,
                "suggested_action": "请选择更小的分子或使用经典算法"
            }
        
        # 6. 获取或计算电子数
        if electrons is None:
            electrons = dataset.molecule.n_electrons
            full_config["parameters_used"]["electrons"] = electrons
        
        # 7. 验证电子数合理性
        if electrons <= 0:
            return {
                "success": False,
                "error": f"电子数无效: {electrons}",
                "algorithm_config": full_config,
                "suggested_action": "请提供正确的电子数"
            }
        
        # 8. 创建量子设备
        dev = qml.device("lightning.qubit", wires=qubits)
        
        # 9. 生成Hartree-Fock初态
        hf = qml.qchem.hf_state(electrons, qubits)
        
        # 10. 定义量子电路
        @qml.qnode(dev, interface="jax")
        def circuit(param, wires):
            qml.BasisState(hf, wires=wires)
            # 根据量子比特数调整DoubleExcitation
            if qubits >= 4:
                qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
            else:
                # 对于更少量子比特的情况，使用SingleExcitation
                qml.SingleExcitation(param, wires=[0, 1])
            return qml.expval(H)
        
        # 11. 定义代价函数
        def cost_fn(param):
            return circuit(param, wires=range(qubits))
        
        # 12. 选择和配置优化器
        if optimizer_type == 'sgd':
            opt = optax.sgd(learning_rate=learning_rate)
        elif optimizer_type == 'adam':
            opt = optax.adam(learning_rate=learning_rate)
        elif optimizer_type == 'adagrad':
            opt = optax.adagrad(learning_rate=learning_rate)
        else:
            opt = optax.sgd(learning_rate=learning_rate)
            full_config["parameters_used"]["optimizer"] = 'sgd'  # 修正为实际使用的
        
        # 13. 初始化优化参数
        theta = np.array(0.0)
        energy_history = []
        initial_energy = cost_fn(theta)
        energy_history.append(float(initial_energy))
        
        opt_state = opt.init(theta)
        
        # 14. 执行优化循环
        converged = False
        iteration_count = 0
        
        for n in range(max_iterations):
            # 计算梯度并更新参数
            gradient = jax.grad(cost_fn)(theta)
            updates, opt_state = opt.update(gradient, opt_state)
            theta = optax.apply_updates(theta, updates)
            
            # 计算新能量
            current_energy = cost_fn(theta)
            energy_history.append(float(current_energy))
            
            # 检查收敛
            conv = np.abs(energy_history[-1] - energy_history[-2])
            iteration_count = n + 1
            
            if conv <= conv_tol:
                converged = True
                break
        
        # 15. 构建返回结果
        final_energy = energy_history[-1]
        
        result = {
            "success": True,
            "molecule": molecule,
            "ground_state_energy": final_energy,
            "energy_unit": "Hartree",
            
            "algorithm_config": full_config,
            
            "optimization_info": {
                "converged": converged,
                "iterations_performed": iteration_count,
                "max_iterations_allowed": max_iterations,
                "final_parameter": float(theta),
                "initial_energy": energy_history[0],
                "final_energy": final_energy,
                "energy_improvement": energy_history[0] - final_energy,
                "convergence_threshold": conv_tol,
                "final_gradient_norm": float(np.abs(conv)) if iteration_count > 0 else 0.0
            },
            
            "system_info": {
                "qubits_used": qubits,
                "electrons": electrons,
                "hamiltonian_terms": len(H.ops) if hasattr(H, 'ops') else "unknown",
                "basis_set": "minimal",
                "hartree_fock_state": hf.tolist()
            },
            
            "computational_details": {
                "device_type": "lightning.qubit",
                "backend": "jax",
                "precision": "float64",
                "ansatz_type": "DoubleExcitation" if qubits >= 4 else "SingleExcitation"
            }
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"VQE计算过程中发生错误: {str(e)}",
            "molecule": molecule,
            "algorithm_config": full_config,
            "suggested_action": "请检查输入参数或联系技术支持"
        }


def validate_vqe_config(config):
    """
    验证VQE配置参数的辅助函数
    
    Args:
        config (dict): 配置字典
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(config, dict):
        return False, "配置必须是字典格式"
    
    if 'molecule' not in config:
        return False, "缺少必需参数: molecule"
    
    user_config = config.get('config', {})
    
    # 验证数值参数范围
    if 'max_iterations' in user_config:
        if not isinstance(user_config['max_iterations'], int) or user_config['max_iterations'] < 1:
            return False, "max_iterations必须是正整数"
    
    if 'learning_rate' in user_config:
        lr = user_config['learning_rate']
        if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
            return False, "learning_rate必须在(0, 1]范围内"
    
    if 'optimizer' in user_config:
        if user_config['optimizer'] not in ['sgd', 'adam', 'adagrad']:
            return False, "optimizer必须是sgd, adam或adagrad之一"
    
    return True, ""


# 估算函数
def estimate_vqe_resources(molecule):
    """
    估算VQE算法所需资源
    
    Args:
        molecule (str): 分子标识符
        
    Returns:
        dict: 资源估算结果
    """
    try:
        dataset = qml.data.load('qchem', molname=molecule)[0]
        qubits = len(dataset.hamiltonian.wires)
        electrons = dataset.molecule.n_electrons
        
        return {
            "qubits_required": qubits,
            "electrons": electrons,
            "feasible": qubits <= 12,
            "estimated_time": "几分钟" if qubits <= 8 else "较长时间"
        }
    except:
        return {
            "qubits_required": "unknown",
            "electrons": "unknown", 
            "feasible": False,
            "error": f"无法分析分子: {molecule}"
        }