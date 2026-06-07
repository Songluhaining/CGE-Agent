from evoagentx.core.module import BaseModule
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pydantic import Field, PrivateAttr


# 假设 BaseModule 在你的项目中可用，这里模拟引用
# from .base_module import BaseModule

# 为了演示完整性，这里提供一个最小化的 BaseModule 桩
# class BaseModule:
#     def __init__(self, **kwargs):
#         for k, v in kwargs.items(): setattr(self, k, v)
#
#     def dict(self): return self.__dict__


class GraphVariable(BaseModule):
    """
    变量节点：代表数据状态 OR 步骤健康状态
    取值范围：0 (False/错误/故障) 或 1 (True/正确/正常)
    """
    name: str
    is_observed: bool = False
    observed_value: Optional[float] = None  # in [0, 1], supports soft evidence

    # 用于区分这是“数据流节点”还是“步骤健康节点”
    is_health_node: bool = False

    # 存储信念 [P(Value=0), P(Value=1)]
    # 对于健康节点，初始先验通常设为高概率正常 (e.g., [0.01, 0.99])
    # 对于数据节点，初始先验通常设为中立 (e.g., [0.5, 0.5])
    _belief: np.ndarray = PrivateAttr(default_factory=lambda: np.array([0.5, 0.5]))

    # 存储传入的消息 {edge_id: message_array}
    _incoming_messages: Dict[str, np.ndarray] = PrivateAttr(default_factory=dict)

    _prior: np.ndarray = PrivateAttr(default_factory=lambda: np.array([0.5, 0.5]))
    _belief: np.ndarray = PrivateAttr(default_factory=lambda: np.array([0.5, 0.5]))

    def set_prior(self, prob_true: float):
        self._prior = np.array([1.0 - prob_true, prob_true])
        self._belief = self._prior.copy()

    def get_belief_prob(self) -> float:
        """获取当前该节点为 True (正常/正确) 的概率"""
        # 如果被观测，直接返回观测值
        if self.is_observed:
            return float(self.observed_value)
        return self._belief[1]

    def reset_messages(self):
        self._incoming_messages.clear()


class GraphEdge(BaseModule):
    """
    连接变量和因子的边。
    存储双向消息。
    """
    id: str
    variable_name: str
    factor_name: str

    # 消息：变量 -> 因子
    _msg_v2f: np.ndarray = PrivateAttr(default_factory=lambda: np.array([0.5, 0.5]))
    # 消息：因子 -> 变量
    _msg_f2v: np.ndarray = PrivateAttr(default_factory=lambda: np.array([0.5, 0.5]))

    def reset(self):
        self._msg_v2f = np.array([0.5, 0.5])
        self._msg_f2v = np.array([0.5, 0.5])
