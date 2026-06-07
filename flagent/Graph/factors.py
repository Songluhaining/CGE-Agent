import string

import numpy as np
from typing import List, Dict

from flagent.Graph.schema import BaseModule


# from .schema import BaseModule

def _safe_normalize(msg: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    msg = np.asarray(msg, dtype=float)

    if not np.all(np.isfinite(msg)):
        return np.array([0.5, 0.5], dtype=float)

    msg = np.maximum(msg, eps)
    s = np.sum(msg)
    if s <= 0:
        return np.array([0.5, 0.5], dtype=float)
    return msg / s

class GraphFactor(BaseModule):
    """因子基类"""
    name: str
    # 连接的变量名列表
    connected_vars: List[str] = []

    def compute_message_to_variable(self, target_var_name: str,
                                    incoming_messages: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class StepLogicFactor(GraphFactor):
    """
    步骤逻辑因子。
    连接三个变量：[Input_Data, Health_Status, Output_Data]
    """
    input_var: str
    health_var: str
    output_var: str

    # 噪声参数：即使输入和健康都正常，产生错误的概率 (偶然错误)
    slip_prob: float = 0.01
    # 猜对参数：即使输入错误，输出碰巧正确的概率 (极低)
    guess_prob: float = 0.001

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connected_vars = [self.input_var, self.health_var, self.output_var]

    def _get_potential_matrix(self) -> np.ndarray:
        """
        构建 3D 势函数矩阵 (Conditional Probability Table)
        维度顺序: [Input(i), Health(h), Output(o)]
        Shape: (2, 2, 2)
        """
        potential = np.zeros((2, 2, 2))

        # 遍历所有状态 (0=False, 1=True)
        for i in [0, 1]:  # Input
            for h in [0, 1]:  # Health
                # 计算 P(Output=1 | Input=i, Health=h)
                prob_output_true = 0.0

                if h == 0:
                    # 如果步骤本身坏了，输出几乎必然是坏的 (Output=1 的概率极低)
                    prob_output_true = self.guess_prob
                elif i == 0:
                    # 如果步骤是好的，但输入是坏的，输出也是坏的 (Garbage In Garbage Out)
                    prob_output_true = self.guess_prob
                else:
                    # 步骤好 + 输入好 = 输出好 (减去一点点偶然滑落概率)
                    prob_output_true = 1.0 - self.slip_prob

                potential[i, h, 1] = prob_output_true
                potential[i, h, 0] = 1.0 - prob_output_true

        return potential

    def compute_message_to_variable(self, target_var_name: str,
                                    incoming_messages: Dict[str, np.ndarray]) -> np.ndarray:
        """
        标准 Sum-Product 算法实现：
        msg(f->v) = Sum_{except v} ( Potential * Product(msg_others) )
        """
        # 1. 获取输入消息，如果没有则默认为均匀分布
        msg_input = incoming_messages.get(self.input_var, np.array([0.5, 0.5]))
        msg_health = incoming_messages.get(self.health_var, np.array([0.5, 0.5]))
        msg_output = incoming_messages.get(self.output_var, np.array([0.5, 0.5]))

        # 2. 获取势函数 (2,2,2) 对应 (Input, Health, Output)
        potential = self._potential_cache

        # 3. 计算边缘化求和 (Marginalization)
        # 利用 numpy 的 einsum (爱因斯坦求和约定) 进行高效张量运算

        if target_var_name == self.output_var:
            # 计算发给 Output 的消息： Sum_{input, health} (P * msg_i * msg_h)
            # 公式解析: result[k] = sum_{i,j} potential[i,j,k] * msg_input[i] * msg_health[j]
            result = np.einsum('ijk,i,j->k', potential, msg_input, msg_health)

        elif target_var_name == self.input_var:
            # 计算发给 Input 的消息 (反向推理)： Sum_{health, output} (P * msg_h * msg_o)
            # 公式解析: result[i] = sum_{j,k} potential[i,j,k] * msg_health[j] * msg_output[k]
            result = np.einsum('ijk,j,k->i', potential, msg_health, msg_output)

        elif target_var_name == self.health_var:
            # 计算发给 Health 的消息 (根因定位核心)： Sum_{input, output} (P * msg_i * msg_o)
            # 公式解析: result[j] = sum_{i,k} potential[i,j,k] * msg_input[i] * msg_output[k]
            result = np.einsum('ijk,i,k->j', potential, msg_input, msg_output)
        else:
            raise ValueError(f"Target variable {target_var_name} not connected to factor {self.name}")

        # 4. 归一化以防止数值溢出/下溢
        return _safe_normalize(result)

class MultiInputStepFactor(GraphFactor):
    """
    支持多输入的工作流步骤因子。
    连接：[N个 Input_Data] + [1个 Health_Status] -> [1个 Output_Data]
    """
    input_vars: List[str]
    health_var: str
    output_var: str

    slip_prob: float = 0.01  # 偶发错误概率 (一切正常但输出错误)
    guess_prob: float = 0.001  # 蒙对概率 (有故障但碰巧输出正确)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connected_vars = self.input_vars + [self.health_var, self.output_var]
        self._potential_cache = self._get_potential_matrix()

    def _get_potential_matrix(self) -> np.ndarray:
        """
        动态构建多维势函数矩阵。
        维度顺序: [Input_1, ..., Input_N, Health, Output]
        """
        num_inputs = len(self.input_vars)
        shape = tuple([2] * num_inputs + [2, 2])
        potential = np.zeros(shape)

        # 遍历所有可能的状态组合
        for idx in np.ndindex(*shape):
            inputs_state = idx[:num_inputs]
            health_state = idx[-2]
            output_state = idx[-1]

            # 逻辑：必须所有输入都为 1 (正确)
            all_inputs_good = all(state == 1 for state in inputs_state)

            if health_state == 0 or not all_inputs_good:
                prob_out_true = self.guess_prob
            else:
                prob_out_true = 1.0 - self.slip_prob

            if output_state == 1:
                potential[idx] = prob_out_true
            else:
                potential[idx] = 1.0 - prob_out_true

        return potential

    def compute_message_to_variable(self, target_var_name: str,
                                    incoming_messages: Dict[str, np.ndarray]) -> np.ndarray:
        """使用 numpy.einsum 进行高维张量的边缘化求和"""
        num_inputs = len(self.input_vars)
        potential = self._potential_cache

        # 动态生成 einsum 下标字母 (a, b, c... 用于输入, y 用于健康, z 用于输出)
        letters = string.ascii_lowercase
        in_chars = letters[:num_inputs]
        h_char = letters[num_inputs]
        o_char = letters[num_inputs + 1]

        # 按维度顺序准备所有传入消息
        msgs = []
        for inv in self.input_vars:
            msgs.append(incoming_messages.get(inv, np.array([0.5, 0.5])))
        msgs.append(incoming_messages.get(self.health_var, np.array([0.5, 0.5])))
        msgs.append(incoming_messages.get(self.output_var, np.array([0.5, 0.5])))

        # 根据目标变量，构造求和公式
        if target_var_name == self.output_var:
            # 目标是输出：对所有输入和健康状态求和
            subscript = f"{in_chars}{h_char}{o_char},{','.join(in_chars)},{h_char}->{o_char}"
            operands = [potential] + msgs[:-1]
            result = np.einsum(subscript, *operands)

        elif target_var_name == self.health_var:
            # 目标是健康节点：对所有输入和输出求和
            subscript = f"{in_chars}{h_char}{o_char},{','.join(in_chars)},{o_char}->{h_char}"
            operands = [potential] + msgs[:-2] + [msgs[-1]]
            result = np.einsum(subscript, *operands)

        else:
            # 目标是某个特定的输入节点 (反向传播)
            target_idx = self.input_vars.index(target_var_name)
            target_char = in_chars[target_idx]

            subscript_parts = [f"{in_chars}{h_char}{o_char}"]
            operands = [potential]

            # 拼装除了目标输入之外的所有操作数
            for i, _ in enumerate(self.input_vars):
                if i != target_idx:
                    subscript_parts.append(in_chars[i])
                    operands.append(msgs[i])

            subscript_parts.append(h_char)
            operands.append(msgs[-2])
            subscript_parts.append(o_char)
            operands.append(msgs[-1])

            subscript = f"{','.join(subscript_parts)}->{target_char}"
            result = np.einsum(subscript, *operands)

        # 归一化
        return _safe_normalize(result)

class HealthAggregationFactor(GraphFactor):
    """
    把多个子部件健康聚合成 HealthStep:
      component_vars (Prompt/Params/Parser/Validator) -> output_var (HealthStep)

    逻辑：若所有 component=1，则 HealthStep=1 的概率 ~ 1-slip_prob
         否则 HealthStep=1 的概率 ~ guess_prob
    """
    component_vars: List[str]
    output_var: str
    slip_prob: float = 0.01
    guess_prob: float = 0.001
    failure_strength: float = 0.40

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connected_vars = list(self.component_vars) + [self.output_var]
        self._potential_cache = self._get_potential()

    def _get_potential(self) -> np.ndarray:
        n = len(self.component_vars)
        shape = tuple([2] * n + [2])  # components..., Output
        pot = np.zeros(shape)

        for idx in np.ndindex(*shape):
            comps = idx[:n]
            out = idx[-1]
            healthy_scale = 1.0
            for v in comps:
                if v == 0:
                    healthy_scale *= (1.0 - self.failure_strength)
            p_out_true = self.guess_prob + max(0.0, 1.0 - self.slip_prob - self.guess_prob) * healthy_scale
            p_out_true = min(max(p_out_true, self.guess_prob), 1.0 - self.slip_prob)
            pot[idx] = p_out_true if out == 1 else (1.0 - p_out_true)
        return pot

    def compute_message_to_variable(self, target_var_name: str, incoming_messages: Dict[str, np.ndarray]) -> np.ndarray:
        pot = self._potential_cache
        n = len(self.component_vars)

        letters = string.ascii_lowercase
        comp_chars = letters[:n]
        out_char = letters[n]

        msgs = [incoming_messages.get(v, np.array([0.5, 0.5])) for v in self.component_vars]
        msg_out = incoming_messages.get(self.output_var, np.array([0.5, 0.5]))

        if target_var_name == self.output_var:
            # sum over components
            sub = f"{comp_chars}{out_char}," + ",".join(comp_chars) + f"->{out_char}"
            result = np.einsum(sub, pot, *msgs)
        else:
            # message to one component: sum over other components and output
            t_idx = self.component_vars.index(target_var_name)
            t_char = comp_chars[t_idx]

            subs = [f"{comp_chars}{out_char}"]
            ops = [pot]

            for i, v in enumerate(self.component_vars):
                if i != t_idx:
                    subs.append(comp_chars[i])
                    ops.append(msgs[i])

            subs.append(out_char)
            ops.append(msg_out)

            sub = ",".join(subs) + f"->{t_char}"
            result = np.einsum(sub, *ops)

        return _safe_normalize(result)

class UnaryObservationFactor(GraphFactor):
    """
    只连接两个变量：[Health, Obs]

    用于接入静态观测，比如 prompt 设计质量、参数配置质量。
    """

    health_var: str
    obs_var: str
    good_match_prob: float = 0.98   # health=1 时，obs=1 的概率
    bad_match_prob: float = 0.20    # health=0 时，obs=1 的概率

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connected_vars = [self.health_var, self.obs_var]

    def _get_potential(self) -> np.ndarray:
        # dims: [H, O]
        pot = np.zeros((2, 2))
        for h in [0, 1]:
            p_obs_true = self.good_match_prob if h == 1 else self.bad_match_prob
            pot[h, 1] = p_obs_true
            pot[h, 0] = 1.0 - p_obs_true
        return pot

    def compute_message_to_variable(self, target_var_name: str, incoming_messages: Dict[str, np.ndarray]) -> np.ndarray:
        msg_h = incoming_messages.get(self.health_var, np.array([0.5, 0.5]))
        msg_o = incoming_messages.get(self.obs_var, np.array([0.5, 0.5]))
        pot = self._get_potential()

        if target_var_name == self.obs_var:
            result = np.einsum("ho,h->o", pot, msg_h)
        elif target_var_name == self.health_var:
            result = np.einsum("ho,o->h", pot, msg_o)
        else:
            raise ValueError(f"Target variable {target_var_name} not connected to {self.name}")

        norm = np.sum(result)
        return (result / norm) if norm > 0 else np.array([0.5, 0.5])


class DataObservationFactor(GraphFactor):
    """
    Connects a latent data variable to a soft observed metric variable.
    """

    data_var: str
    obs_var: str
    good_match_prob: float = 0.90
    bad_match_prob: float = 0.10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connected_vars = [self.data_var, self.obs_var]

    def _get_potential(self) -> np.ndarray:
        pot = np.zeros((2, 2))
        for d in [0, 1]:
            p_obs_true = self.good_match_prob if d == 1 else self.bad_match_prob
            pot[d, 1] = p_obs_true
            pot[d, 0] = 1.0 - p_obs_true
        return pot

    def compute_message_to_variable(self, target_var_name: str, incoming_messages: Dict[str, np.ndarray]) -> np.ndarray:
        msg_d = incoming_messages.get(self.data_var, np.array([0.5, 0.5]))
        msg_o = incoming_messages.get(self.obs_var, np.array([0.5, 0.5]))
        pot = self._get_potential()

        if target_var_name == self.obs_var:
            result = np.einsum("do,d->o", pot, msg_d)
        elif target_var_name == self.data_var:
            result = np.einsum("do,o->d", pot, msg_o)
        else:
            raise ValueError(f"Target variable {target_var_name} not connected to {self.name}")

        norm = np.sum(result)
        return (result / norm) if norm > 0 else np.array([0.5, 0.5])


# =========================
# ✅ 新增：局部观测因子
# =========================
class ObservationFactor(GraphFactor):
    """
    连接三个变量：[Health, DataOut, Obs]
    用于把“解析/校验是否通过”作为观测注入图中。

    若 Health=1（该部件健康），Obs 与 DataOut 强一致（1-slip_prob）
    若 Health=0（该部件不健康），Obs 与 DataOut 弱一致（bad_match_prob，默认0.5等于随机）
    """
    health_var: str
    data_var: str
    obs_var: str
    slip_prob: float = 0.02
    bad_match_prob: float = 0.5  # Health坏时，Obs==Data 的概率（0.5 表示完全没信息）

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connected_vars = [self.health_var, self.data_var, self.obs_var]

    def _get_potential(self) -> np.ndarray:
        # dims: [H, D, O]
        pot = np.zeros((2, 2, 2))
        for h in [0, 1]:
            for d in [0, 1]:
                # P(O==D | H)
                p_match = (1.0 - self.slip_prob) if h == 1 else self.bad_match_prob
                for o in [0, 1]:
                    pot[h, d, o] = p_match if (o == d) else (1.0 - p_match)
        return pot

    def compute_message_to_variable(self, target_var_name: str, incoming_messages: Dict[str, np.ndarray]) -> np.ndarray:
        msg_h = incoming_messages.get(self.health_var, np.array([0.5, 0.5]))
        msg_d = incoming_messages.get(self.data_var, np.array([0.5, 0.5]))
        msg_o = incoming_messages.get(self.obs_var, np.array([0.5, 0.5]))
        pot = self._get_potential()

        if target_var_name == self.obs_var:
            # sum_{h,d} pot[h,d,o] * msg_h[h] * msg_d[d]
            result = np.einsum("hdo,h,d->o", pot, msg_h, msg_d)
        elif target_var_name == self.data_var:
            # sum_{h,o} pot[h,d,o] * msg_h[h] * msg_o[o]
            result = np.einsum("hdo,h,o->d", pot, msg_h, msg_o)
        elif target_var_name == self.health_var:
            # sum_{d,o} pot[h,d,o] * msg_d[d] * msg_o[o]
            result = np.einsum("hdo,d,o->h", pot, msg_d, msg_o)
        else:
            raise ValueError(f"Target variable {target_var_name} not connected to {self.name}")

        norm = np.sum(result)
        return (result / norm) if norm > 0 else np.array([0.5, 0.5])

class HealthGatedStepFactor(GraphFactor):
    """
    二元步骤因子：Health -> Data。
    用于步骤因果链中不存在上游数据输入、健康状态直接决定数据状态的场景（例如 Params 分支）。
        P(Data=1 | Health=1) = 1 - slip_prob
        P(Data=1 | Health=0) = guess_prob
    """
    health_var: str
    data_var: str
    slip_prob: float = 0.05
    guess_prob: float = 0.001

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connected_vars = [self.health_var, self.data_var]

    def _get_potential(self) -> np.ndarray:
        pot = np.zeros((2, 2))
        for h in [0, 1]:
            p_true = (1.0 - self.slip_prob) if h == 1 else self.guess_prob
            pot[h, 1] = p_true
            pot[h, 0] = 1.0 - p_true
        return pot

    def compute_message_to_variable(self, target_var_name: str,
                                    incoming_messages: Dict[str, np.ndarray]) -> np.ndarray:
        msg_h = incoming_messages.get(self.health_var, np.array([0.5, 0.5]))
        msg_d = incoming_messages.get(self.data_var, np.array([0.5, 0.5]))
        pot = self._get_potential()

        if target_var_name == self.data_var:
            result = np.einsum("hd,h->d", pot, msg_h)
        elif target_var_name == self.health_var:
            result = np.einsum("hd,d->h", pot, msg_d)
        else:
            raise ValueError(f"Target variable {target_var_name} not connected to {self.name}")
        return _safe_normalize(result)

