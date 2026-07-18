# models/__init__.py
from .baseline import EntangledBaselineModel
from .proposed import DisentangledProposedModel

# 显式定义外界 import * 时允许访问的类
__all__ = [
    'EntangledBaselineModel',
    'DisentangledProposedModel'
]