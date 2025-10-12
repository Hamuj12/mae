# util/compat.py
from math import inf as _inf
try:
    # PyTorch 1.x path (legacy)
    from torch._six import container_abcs as _container_abcs  # type: ignore
    from torch._six import string_classes as _string_classes   # type: ignore
    from torch._six import int_classes as _int_classes         # type: ignore
    from torch._six import inf as _pt_inf                      # type: ignore
    inf = _pt_inf
    container_abcs = _container_abcs
    string_classes = _string_classes
    int_classes = _int_classes
except Exception:
    # PyTorch 2.x path
    from collections import abc as container_abcs  # Iterable, Mapping, Sequence live here
    inf = _inf
    string_classes = (str,)
    int_classes = (int,)