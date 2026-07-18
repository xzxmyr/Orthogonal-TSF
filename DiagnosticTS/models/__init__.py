# models/__init__.py
from .baseline import EntangledBaselineModel
from .proposed import DiagnosticTS  # 🆕 路由指向全新重构的物理哈达玛乘性主模型

__all__ = [
    'EntangledBaselineModel',
    'DiagnosticTS'
]