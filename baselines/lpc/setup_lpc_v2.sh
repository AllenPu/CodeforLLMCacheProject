#!/bin/bash

# LPC-v2 一键安装脚本
# 使用方法: bash setup_lpc_v2.sh [LPC_ROOT_PATH]
# 例如: bash setup_lpc_v2.sh /home/ziang/LPC

set -e

# 获取 LPC 根目录
LPC_ROOT="${1:-$HOME/LPC}"
TARGET_DIR="$LPC_ROOT/vllm/vllm/core/lpc_v2"

echo "=========================================="
echo "LPC-v2 安装脚本"
echo "目标目录: $TARGET_DIR"
echo "=========================================="

# 创建目录
mkdir -p "$TARGET_DIR"
echo "✓ 创建目录: $TARGET_DIR"

# ==================== 文件 1: __init__.py ====================
cat > "$TARGET_DIR/__init__.py" << 'ENDOFFILE'
"""
LPC-v2: 可学习的 Scale 函数 + 上下文相关的 Token 权重

改进点：
1. Scale 函数：从固定值变为可学习的 f(elapsed_time)
2. Token 权重：从人工设定变为可学习的 g(context, token_type)
"""

from .token_types import TokenType, TokenTypeTracker, detect_cot_ranges
from .scale_function import LearnableScaleFunction
from .token_weight import ContextAwareTokenWeight
from .bucket_classifier import BucketClassifier

__all__ = [
    'TokenType',
    'TokenTypeTracker',
    'detect_cot_ranges',
    'LearnableScaleFunction',
    'ContextAwareTokenWeight',
    'BucketClassifier',
]

__version__ = '0.1.0'
ENDOFFILE
echo "✓ 创建文件: __init__.py"

# ==================== 文件 2: token_types.py ====================
cat > "$TARGET_DIR/token_types.py" << 'ENDOFFILE'
"""
Token 类型定义和追踪器

Token 类型：
- SYSTEM_PROMPT: System 指令，复用率最高
- USER_PROMPT: 用户输入，多轮对话中会重复
- TOOL_OUTPUT: 工具返回结果
- RESPONSE: 最终回答
- COT: Chain-of-Thought，<think>...</think> 内的内容，复用率最低
"""

from enum import Enum
from typing import List, Tuple, Optional


class TokenType(Enum):
    """Token 类型枚举"""
    SYSTEM_PROMPT = 0
    USER_PROMPT = 1
    TOOL_OUTPUT = 2
    RESPONSE = 3
    COT = 4


# Token 类型名称映射
TOKEN_TYPE_NAMES = {
    TokenType.SYSTEM_PROMPT: "system_prompt",
    TokenType.USER_PROMPT: "user_prompt",
    TokenType.TOOL_OUTPUT: "tool_output",
    TokenType.RESPONSE: "response",
    TokenType.COT: "cot",
}

# 反向映射
NAME_TO_TOKEN_TYPE = {v: k for k, v in TOKEN_TYPE_NAMES.items()}


class TokenTypeTracker:
    """
    Token 类型追踪器（状态机）
    
    用于在生成过程中追踪当前 token 属于哪种类型
    """
    
    def __init__(self):
        self.in_cot = False
        self.current_role = None
    
    def reset(self):
        """重置状态"""
        self.in_cot = False
        self.current_role = None
    
    def set_role(self, role: str):
        """设置当前消息的角色"""
        self.current_role = role
        if role != "assistant":
            self.in_cot = False
    
    def get_token_type(
        self, 
        token_content: str, 
        message_role: Optional[str] = None
    ) -> TokenType:
        """
        根据 token 内容和消息角色判断类型
        
        Args:
            token_content: token 的文本内容
            message_role: 当前消息的角色（system/user/assistant/tool）
                         如果为 None，使用上次设置的 role
        
        Returns:
            TokenType: token 的类型
        """
        if message_role is not None:
            self.current_role = message_role
        
        role = self.current_role
        
        # 根据消息角色判断
        if role == "system":
            return TokenType.SYSTEM_PROMPT
        elif role == "user":
            return TokenType.USER_PROMPT
        elif role == "tool":
            return TokenType.TOOL_OUTPUT
        
        # 对于 assistant 消息，区分 CoT 和 Response
        if role == "assistant":
            # 检测 <think> 标记
            if "<think>" in token_content or "<|think|>" in token_content:
                self.in_cot = True
                return TokenType.COT
            elif "</think>" in token_content or "<|/think|>" in token_content:
                self.in_cot = False
                return TokenType.COT  # 闭合标签本身也是 CoT
            
            return TokenType.COT if self.in_cot else TokenType.RESPONSE
        
        # 默认返回 RESPONSE
        return TokenType.RESPONSE
    
    def classify_tokens(
        self, 
        tokens: List[str], 
        message_role: str
    ) -> List[TokenType]:
        """
        批量分类 tokens
        
        Args:
            tokens: token 文本列表
            message_role: 消息角色
        
        Returns:
            List[TokenType]: 每个 token 的类型
        """
        self.set_role(message_role)
        return [self.get_token_type(t) for t in tokens]


def detect_cot_ranges(
    token_ids: List[int], 
    think_start_id: int, 
    think_end_id: int
) -> List[Tuple[int, int]]:
    """
    检测 CoT token 的范围（基于 token ID）
    
    Args:
        token_ids: token ID 列表
        think_start_id: <think> 的 token ID
        think_end_id: </think> 的 token ID
    
    Returns:
        List[Tuple[int, int]]: CoT 范围列表 [(start, end), ...]
    """
    cot_ranges = []
    in_cot = False
    cot_start = 0
    
    for i, tid in enumerate(token_ids):
        if tid == think_start_id:
            in_cot = True
            cot_start = i
        elif tid == think_end_id:
            if in_cot:
                cot_ranges.append((cot_start, i + 1))
                in_cot = False
    
    # 处理未闭合的情况（模型还在生成 CoT）
    if in_cot:
        cot_ranges.append((cot_start, len(token_ids)))
    
    return cot_ranges


def detect_cot_ranges_by_text(
    text: str,
    start_marker: str = "<think>",
    end_marker: str = "</think>"
) -> List[Tuple[int, int]]:
    """
    检测 CoT 的字符范围（基于文本）
    
    Args:
        text: 完整文本
        start_marker: 开始标记
        end_marker: 结束标记
    
    Returns:
        List[Tuple[int, int]]: 字符位置范围列表
    """
    cot_ranges = []
    start = 0
    
    while True:
        think_start = text.find(start_marker, start)
        if think_start == -1:
            break
        
        think_end = text.find(end_marker, think_start)
        if think_end == -1:
            # 未闭合，假设到末尾都是 CoT
            cot_ranges.append((think_start, len(text)))
            break
        
        cot_ranges.append((think_start, think_end + len(end_marker)))
        start = think_end + len(end_marker)
    
    return cot_ranges


def get_dominant_token_type(token_types: List[TokenType]) -> TokenType:
    """
    获取主要的 token 类型（众数）
    
    Args:
        token_types: token 类型列表
    
    Returns:
        TokenType: 出现最多的类型
    """
    if not token_types:
        return TokenType.RESPONSE
    
    from collections import Counter
    counter = Counter(token_types)
    return counter.most_common(1)[0][0]
ENDOFFILE
echo "✓ 创建文件: token_types.py"

# ==================== 文件 3: scale_function.py ====================
cat > "$TARGET_DIR/scale_function.py" << 'ENDOFFILE'
"""
可学习的 Scale 函数

原始 LPC 使用固定的 scale = 0.01
改进：scale = f(elapsed_time)，通过 MLP 学习

特点：
1. 输出保证 > 0（使用 Softplus）
2. 残差连接（base_scale + learned_delta）
3. 输入归一化提高训练稳定性
"""

import torch
import torch.nn as nn
import math
from typing import Union


class LearnableScaleFunction(nn.Module):
    """
    可学习的 Scale 函数
    
    scale = base_scale + MLP(elapsed_time / time_scale)
    
    Args:
        hidden_dim: MLP 隐藏层维度
        num_layers: MLP 层数
        time_scale: 时间归一化因子（典型值：500秒）
        base_scale_init: 基础 scale 初始值
    """
    
    def __init__(
        self, 
        hidden_dim: int = 32, 
        num_layers: int = 2,
        time_scale: float = 500.0,
        base_scale_init: float = 0.01
    ):
        super().__init__()
        
        self.time_scale = time_scale
        
        # 构建 MLP
        layers = []
        input_dim = 1
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Softplus(beta=1.0))  # 保证输出 > 0
        
        self.mlp = nn.Sequential(*layers)
        
        # 可学习的基础 scale（残差连接）
        self.base_scale = nn.Parameter(torch.tensor(base_scale_init))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，使初始输出接近 0（依赖 base_scale）"""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, elapsed_time: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        计算 scale 值
        
        Args:
            elapsed_time: 距离上次访问的时间（秒）
                         可以是标量、1D tensor 或 batch
        
        Returns:
            scale: 与输入同形状的 scale 值
        """
        # 转换为 tensor
        if not isinstance(elapsed_time, torch.Tensor):
            elapsed_time = torch.tensor(elapsed_time, dtype=torch.float32)
        
        # 记录原始形状
        original_shape = elapsed_time.shape
        original_dim = elapsed_time.dim()
        
        # 归一化输入
        t_normalized = elapsed_time / self.time_scale
        
        # 调整维度为 (batch, 1)
        if t_normalized.dim() == 0:
            t_normalized = t_normalized.unsqueeze(0).unsqueeze(-1)
        elif t_normalized.dim() == 1:
            t_normalized = t_normalized.unsqueeze(-1)
        
        # MLP 前向传播
        scale_delta = self.mlp(t_normalized).squeeze(-1)
        
        # 加上基础 scale
        scale = self.base_scale + scale_delta
        
        # 恢复原始形状
        if original_dim == 0:
            scale = scale.squeeze()
        
        return scale
    
    def compute_decay(self, elapsed_time: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        直接计算 decay 值
        
        decay = exp(-elapsed_time * scale)
        """
        if not isinstance(elapsed_time, torch.Tensor):
            elapsed_time = torch.tensor(elapsed_time, dtype=torch.float32)
        
        scale = self.forward(elapsed_time)
        decay = torch.exp(-elapsed_time * scale)
        
        return decay
    
    def get_scale_curve(
        self, 
        max_time: float = 500.0, 
        num_points: int = 100
    ) -> tuple:
        """
        获取 scale 曲线（用于可视化）
        
        Returns:
            (times, scales): numpy arrays
        """
        times = torch.linspace(0, max_time, num_points)
        with torch.no_grad():
            scales = self.forward(times)
        return times.numpy(), scales.numpy()


class FixedScaleFunction(nn.Module):
    """
    固定 Scale 函数（用于对比实验）
    
    与原始 LPC 相同，scale = 常数
    """
    
    def __init__(self, scale: float = 0.01):
        super().__init__()
        self.scale = scale
    
    def forward(self, elapsed_time: Union[float, torch.Tensor]) -> torch.Tensor:
        if not isinstance(elapsed_time, torch.Tensor):
            elapsed_time = torch.tensor(elapsed_time, dtype=torch.float32)
        return torch.full_like(elapsed_time, self.scale)
    
    def compute_decay(self, elapsed_time: Union[float, torch.Tensor]) -> torch.Tensor:
        if not isinstance(elapsed_time, torch.Tensor):
            elapsed_time = torch.tensor(elapsed_time, dtype=torch.float32)
        return torch.exp(-elapsed_time * self.scale)
ENDOFFILE
echo "✓ 创建文件: scale_function.py"

# ==================== 文件 4: token_weight.py ====================
cat > "$TARGET_DIR/token_weight.py" << 'ENDOFFILE'
"""
上下文相关的 Token 权重

原始方案：人工设定固定权重
改进：weight = g(context_embedding, token_type)

特点：
1. 同一 token 类型在不同上下文中权重不同
2. 端到端学习，无需人工设定
3. 支持残差连接（base_weight + learned_delta）
"""

import torch
import torch.nn as nn
from typing import Union, Dict, Optional

from .token_types import TokenType, TOKEN_TYPE_NAMES


class ContextAwareTokenWeight(nn.Module):
    """
    上下文相关的 Token 权重
    
    weight = base_weight[token_type] + MLP(context || token_embedding)
    
    Args:
        context_dim: 上下文 embedding 维度（默认 384，对应 e5-small）
        token_type_dim: token 类型 embedding 维度
        hidden_dim: MLP 隐藏层维度
        num_token_types: token 类型数量
    """
    
    NUM_TOKEN_TYPES = 5  # 与 TokenType 枚举对应
    
    def __init__(
        self,
        context_dim: int = 384,
        token_type_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.context_dim = context_dim
        self.token_type_dim = token_type_dim
        
        # Token 类型 embedding
        self.token_type_embedding = nn.Embedding(
            num_embeddings=self.NUM_TOKEN_TYPES,
            embedding_dim=token_type_dim
        )
        
        # 权重预测网络
        self.weight_net = nn.Sequential(
            nn.Linear(context_dim + token_type_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 保证输出 > 0
        )
        
        # 可学习的基础权重（每种 token type 一个）
        # 初始化：system_prompt 高，cot 低
        initial_weights = torch.tensor([2.0, 1.2, 1.0, 0.8, 0.1])
        self.base_weights = nn.Parameter(initial_weights)
        
        # 初始化网络权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.weight_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        context_embedding: torch.Tensor,
        token_type: Union[int, TokenType, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算 token 权重
        
        Args:
            context_embedding: 上下文 embedding
                              形状: (context_dim,) 或 (batch, context_dim)
            token_type: token 类型
                       可以是 int、TokenType 枚举或 tensor
        
        Returns:
            weight: 权重值，形状与输入 batch 维度一致
        """
        # 处理 token_type 输入
        if isinstance(token_type, TokenType):
            token_type_id = torch.tensor(token_type.value, dtype=torch.long)
        elif isinstance(token_type, int):
            token_type_id = torch.tensor(token_type, dtype=torch.long)
        else:
            token_type_id = token_type.long()
        
        # 确保在正确的设备上
        device = context_embedding.device
        if not isinstance(token_type_id, torch.Tensor):
            token_type_id = torch.tensor(token_type_id, dtype=torch.long, device=device)
        else:
            token_type_id = token_type_id.to(device)
        
        # 处理 context_embedding 维度
        squeeze_output = False
        if context_embedding.dim() == 1:
            context_embedding = context_embedding.unsqueeze(0)
            squeeze_output = True
        
        batch_size = context_embedding.shape[0]
        
        # 扩展 token_type_id 到 batch
        if token_type_id.dim() == 0:
            token_type_id = token_type_id.expand(batch_size)
        
        # 获取 token type embedding
        token_emb = self.token_type_embedding(token_type_id)  # (batch, token_type_dim)
        
        # 拼接 context 和 token type
        combined = torch.cat([context_embedding, token_emb], dim=-1)
        
        # 计算权重增量
        weight_delta = self.weight_net(combined).squeeze(-1)
        
        # 加上基础权重
        base_weight = self.base_weights[token_type_id]
        weight = base_weight + weight_delta
        
        # 恢复原始形状
        if squeeze_output:
            weight = weight.squeeze(0)
        
        return weight
    
    def get_all_weights(
        self, 
        context_embedding: torch.Tensor
    ) -> Dict[str, float]:
        """
        获取所有 token 类型的权重（用于分析）
        
        Args:
            context_embedding: 上下文 embedding
        
        Returns:
            Dict: {token_type_name: weight}
        """
        weights = {}
        with torch.no_grad():
            for token_type in TokenType:
                w = self.forward(context_embedding, token_type)
                name = TOKEN_TYPE_NAMES[token_type]
                weights[name] = w.item() if w.dim() == 0 else w.mean().item()
        return weights
    
    def get_base_weights(self) -> Dict[str, float]:
        """获取基础权重（不考虑上下文）"""
        weights = {}
        for token_type in TokenType:
            name = TOKEN_TYPE_NAMES[token_type]
            weights[name] = self.base_weights[token_type.value].item()
        return weights


class FixedTokenWeight(nn.Module):
    """
    固定 Token 权重（用于对比实验 / 消融实验）
    
    使用人工设定的固定权重，不依赖上下文
    """
    
    # 默认权重（基于经验/统计）
    DEFAULT_WEIGHTS = {
        TokenType.SYSTEM_PROMPT: 2.0,
        TokenType.USER_PROMPT: 1.2,
        TokenType.TOOL_OUTPUT: 1.0,
        TokenType.RESPONSE: 0.8,
        TokenType.COT: 0.1,
    }
    
    def __init__(self, weights: Optional[Dict[TokenType, float]] = None):
        super().__init__()
        
        if weights is None:
            weights = self.DEFAULT_WEIGHTS
        
        # 转换为 tensor
        weight_tensor = torch.zeros(len(TokenType))
        for token_type, weight in weights.items():
            weight_tensor[token_type.value] = weight
        
        self.register_buffer('weights', weight_tensor)
    
    def forward(
        self,
        context_embedding: torch.Tensor,
        token_type: Union[int, TokenType, torch.Tensor]
    ) -> torch.Tensor:
        """
        获取固定权重（忽略 context_embedding）
        """
        if isinstance(token_type, TokenType):
            token_type_id = token_type.value
        elif isinstance(token_type, int):
            token_type_id = token_type
        else:
            token_type_id = token_type.item() if token_type.dim() == 0 else token_type
        
        if isinstance(token_type_id, int):
            return self.weights[token_type_id]
        else:
            return self.weights[token_type_id]
    
    def get_all_weights(self, context_embedding: torch.Tensor = None) -> Dict[str, float]:
        """获取所有权重"""
        return {
            TOKEN_TYPE_NAMES[tt]: self.weights[tt.value].item()
            for tt in TokenType
        }
ENDOFFILE
echo "✓ 创建文件: token_weight.py"

# ==================== 文件 5: bucket_classifier.py ====================
cat > "$TARGET_DIR/bucket_classifier.py" << 'ENDOFFILE'
"""
Bucket 分类器

将对话分类到不同的 bucket（工作负载类型），
每个 bucket 可以有不同的 scale 参数。

Bucket 类型：
- math_reasoning: 数学推理，用户思考时间长
- code_generation: 代码生成，用户需要测试
- creative_writing: 创意写作
- knowledge_qa: 知识问答
- chitchat: 闲聊
- simple_qa: 简单问答，回复快
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union


class BucketClassifier(nn.Module):
    """
    Bucket 分类器
    
    基于 text embedding 分类工作负载类型
    
    Args:
        embedding_dim: 输入 embedding 维度
        hidden_dim: 隐藏层维度
        num_buckets: bucket 数量（默认 6）
    """
    
    # Bucket 定义
    BUCKETS = [
        "math_reasoning",
        "code_generation",
        "creative_writing",
        "knowledge_qa",
        "chitchat",
        "simple_qa"
    ]
    
    # 每个 bucket 对应的默认 scale
    # scale = 1 / 预期平均对话间隔（秒）
    DEFAULT_BUCKET_SCALES = {
        "math_reasoning": 0.005,    # 预期间隔 200 秒
        "code_generation": 0.006,   # 预期间隔 ~167 秒
        "creative_writing": 0.008,  # 预期间隔 ~125 秒
        "knowledge_qa": 0.012,      # 预期间隔 ~83 秒
        "chitchat": 0.015,          # 预期间隔 ~67 秒
        "simple_qa": 0.020,         # 预期间隔 50 秒
    }
    
    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_buckets = len(self.BUCKETS)
        self.embedding_dim = embedding_dim
        
        # 分类网络
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_buckets)
        )
        
        # 可学习的 bucket scales（可选，用于端到端训练）
        default_scales = torch.tensor([
            self.DEFAULT_BUCKET_SCALES[b] for b in self.BUCKETS
        ])
        self.bucket_scales = nn.Parameter(default_scales)
    
    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播，输出 bucket logits
        
        Args:
            text_embedding: (batch, embedding_dim) 或 (embedding_dim,)
        
        Returns:
            logits: (batch, num_buckets) 或 (num_buckets,)
        """
        return self.classifier(text_embedding)
    
    def predict_bucket_id(self, text_embedding: torch.Tensor) -> int:
        """预测 bucket ID"""
        with torch.no_grad():
            logits = self.forward(text_embedding)
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            return torch.argmax(logits).item()
    
    def predict_bucket(self, text_embedding: torch.Tensor) -> str:
        """预测 bucket 名称"""
        bucket_id = self.predict_bucket_id(text_embedding)
        return self.BUCKETS[bucket_id]
    
    def get_scale(self, text_embedding: torch.Tensor) -> float:
        """获取对应的 scale 值"""
        bucket_id = self.predict_bucket_id(text_embedding)
        return self.bucket_scales[bucket_id].item()
    
    def get_scale_by_bucket(self, bucket: Union[str, int]) -> float:
        """根据 bucket 名称或 ID 获取 scale"""
        if isinstance(bucket, str):
            bucket_id = self.BUCKETS.index(bucket)
        else:
            bucket_id = bucket
        return self.bucket_scales[bucket_id].item()
    
    def get_all_scales(self) -> Dict[str, float]:
        """获取所有 bucket 的 scale"""
        return {
            bucket: self.bucket_scales[i].item()
            for i, bucket in enumerate(self.BUCKETS)
        }
    
    @classmethod
    def get_default_scale(cls, bucket: str) -> float:
        """获取默认 scale（类方法）"""
        return cls.DEFAULT_BUCKET_SCALES.get(bucket, 0.01)


class SimpleBucketClassifier:
    """
    简单的 Bucket 分类器（基于关键词，不需要训练）
    
    用于快速实验或作为 baseline
    """
    
    # 关键词映射
    KEYWORDS = {
        "math_reasoning": [
            'calculate', 'solve', 'prove', 'equation', 'math', 'formula',
            '计算', '求解', '证明', '方程', '数学', '公式'
        ],
        "code_generation": [
            'code', 'function', 'implement', 'debug', 'program', 'script',
            '代码', '函数', '实现', '调试', '程序', '脚本', 'python', 'java'
        ],
        "creative_writing": [
            'write', 'story', 'poem', 'creative', 'novel', 'essay',
            '写', '故事', '诗', '创作', '小说', '文章'
        ],
        "knowledge_qa": [
            'what is', 'explain', 'how does', 'why', 'define',
            '什么是', '解释', '为什么', '定义', '介绍'
        ],
        "chitchat": [
            'hello', 'hi', 'how are you', 'chat', 'talk',
            '你好', '聊聊', '闲聊'
        ],
    }
    
    DEFAULT_BUCKET = "simple_qa"
    
    @classmethod
    def classify(cls, text: str) -> str:
        """根据关键词分类"""
        text_lower = text.lower()
        
        for bucket, keywords in cls.KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return bucket
        
        return cls.DEFAULT_BUCKET
    
    @classmethod
    def get_scale(cls, text: str) -> float:
        """获取 scale"""
        bucket = cls.classify(text)
        return BucketClassifier.DEFAULT_BUCKET_SCALES.get(bucket, 0.01)
ENDOFFILE
echo "✓ 创建文件: bucket_classifier.py"

# ==================== 完成 ====================
echo ""
echo "=========================================="
echo "✅ LPC-v2 模块安装完成！"
echo ""
echo "文件列表："
ls -la "$TARGET_DIR"
echo ""
echo "使用方法："
echo "在 evictor.py 中添加："
echo "  from .lpc_v2 import ("
echo "      TokenType,"
echo "      LearnableScaleFunction,"
echo "      ContextAwareTokenWeight,"
echo "      BucketClassifier"
echo "  )"
echo "=========================================="
