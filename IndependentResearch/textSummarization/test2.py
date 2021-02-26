#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x3a6bca0e

# Compiled with Coconut version 1.4.0 [Ernest Scribbler]

"""
   Thus, our first function diagonal_line(n) should
   construct an iterator of all the points, represented as
   coordinate tuples, in the nth diagonal, starting with (0, 0) as the 0th diagonal.
"""

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.abspath(__file__))
_coconut_cached_module = _coconut_sys.modules.get(str("__coconut__"))
if _coconut_cached_module is not None and _coconut_os_path.dirname(_coconut_cached_module.__file__) != _coconut_file_path:
    del _coconut_sys.modules[str("__coconut__")]
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import _coconut, _coconut_NamedTuple, _coconut_MatchError, _coconut_tail_call, _coconut_tco, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_pipe, _coconut_star_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial
from __coconut__ import *
_coconut_sys.path.remove(_coconut_file_path)

# Compiled Coconut: -----------------------------------------------------------


@_coconut_tco
def diagonal_line(n):
    return _coconut_tail_call(map, lambda x: (x, n - x), range(n + 1))

(print)((isinstance)(diagonal_line(0), (list, tuple)))  # False (should be an iterator)
(print)((list)(diagonal_line(0)))  # [(0, 0)]
(print)((list)(diagonal_line(1)))  # [(0, 1), (1, 0)]

"""
linearized_plane should produce an iterator that goes through all the
points in the plane, in order of all the points in the first diagonal(0),
then the second diagonal(1), and so on
"""
@_coconut_tco
def linearized_plane(n=0):
    return _coconut_tail_call(_coconut.itertools.chain.from_iterable, (_coconut_func() for _coconut_func in (lambda: diagonal_line(n), lambda: linearized_plane(n + 1))))

# Note: these tests use $[] notation, which we haven't introduced yet
#  but will introduce later in this case study; for now, just run the
#  tests, and make sure you get the same result as is in the comment
(print)(_coconut_igetitem(linearized_plane(), 0))  # (0, 0)
(print)((list)(_coconut_igetitem(linearized_plane(), _coconut.slice(None, 3))))  # [(0, 0), (0, 1), (1, 0)]

class vector(_coconut.collections.namedtuple("vector", "pts"), _coconut.object):
    """Immutable n-vector."""
    __slots__ = ()
    __ne__ = _coconut.object.__ne__
    def __eq__(self, other):
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)
    def __new__(_cls, *pts):
        return _coconut.tuple.__new__(_cls, pts)
    @_coconut.classmethod
    def _make(cls, iterable, new=_coconut.tuple.__new__, len=None):
        return new(cls, iterable)
    def _asdict(self):
        return _coconut.OrderedDict([("pts", self[:])])
    def __repr__(self):
        return "vector(*pts=%r)" % (self[:],)
    def _replace(_self, **kwds):
        result = self._make(kwds.pop("pts", _self))
        if kwds:
            raise _coconut.ValueError("Got unexpected field names: " + _coconut.repr(kwds.keys()))
        return result
    @_coconut.property
    def pts(self):
        return self[:]
    @_coconut_tco
    def __new__(cls, *pts):
        """Create a new vector from the given pts."""
        _coconut_match_to = pts
        _coconut_match_check = False
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 1) and (_coconut.isinstance(_coconut_match_to[0], vector)):
            v = _coconut_match_to[0]
            _coconut_match_check = True
        if _coconut_match_check:
            return v  # vector(v) where v is a vector should return v
        else:
            return _coconut_tail_call(makedata, cls, *pts)  # accesses base constructor
    @_coconut_tco
    def __abs__(self):
        """Return the magnitude of the vector."""
        return _coconut_tail_call((_coconut_partial(pow, {1: 0.5}, 2)), (sum)(map(_coconut_partial(pow, {1: 2}, 2), self.pts)))
    @_coconut_tco
    def __add__(*_coconut_match_to_args, **_coconut_match_to_kwargs):
        """Add two vectors together."""
        _coconut_match_check = False
        if (_coconut.len(_coconut_match_to_args) == 2) and ("self" not in _coconut_match_to_kwargs) and (_coconut.isinstance(_coconut_match_to_args[1], vector)):
            _coconut_match_temp_0 = _coconut_match_to_args[0] if _coconut.len(_coconut_match_to_args) > 0 else _coconut_match_to_kwargs.pop("self")
            other_pts = _coconut_match_to_args[1][0:]
            if not _coconut_match_to_kwargs:
                self = _coconut_match_temp_0
                _coconut_match_check = True
        if _coconut_match_check and not (len(other_pts) == len(self.pts)):
            _coconut_match_check = False
        if not _coconut_match_check:
            _coconut_match_err = _coconut_MatchError("pattern-matching failed for " "'def __add__(self, vector(*other_pts)                 if len(other_pts) == len(self.pts)) ='" " in " + _coconut.repr(_coconut.repr(_coconut_match_to_args)))
            _coconut_match_err.pattern = 'def __add__(self, vector(*other_pts)                 if len(other_pts) == len(self.pts)) ='
            _coconut_match_err.value = _coconut_match_to_args
            raise _coconut_match_err

        return _coconut_tail_call((vector), *map(_coconut.operator.add, self.pts, other_pts))
    @_coconut_tco
    def __sub__(*_coconut_match_to_args, **_coconut_match_to_kwargs):
        """Subtract one vector from another."""
        _coconut_match_check = False
        if (_coconut.len(_coconut_match_to_args) == 2) and ("self" not in _coconut_match_to_kwargs) and (_coconut.isinstance(_coconut_match_to_args[1], vector)):
            _coconut_match_temp_0 = _coconut_match_to_args[0] if _coconut.len(_coconut_match_to_args) > 0 else _coconut_match_to_kwargs.pop("self")
            other_pts = _coconut_match_to_args[1][0:]
            if not _coconut_match_to_kwargs:
                self = _coconut_match_temp_0
                _coconut_match_check = True
        if _coconut_match_check and not (len(other_pts) == len(self.pts)):
            _coconut_match_check = False
        if not _coconut_match_check:
            _coconut_match_err = _coconut_MatchError("pattern-matching failed for " "'def __sub__(self, vector(*other_pts)                 if len(other_pts) == len(self.pts)) ='" " in " + _coconut.repr(_coconut.repr(_coconut_match_to_args)))
            _coconut_match_err.pattern = 'def __sub__(self, vector(*other_pts)                 if len(other_pts) == len(self.pts)) ='
            _coconut_match_err.value = _coconut_match_to_args
            raise _coconut_match_err

        return _coconut_tail_call((vector), *map(_coconut_minus, self.pts, other_pts))
    @_coconut_tco
    def __neg__(self):
        """Retrieve the negative of the vector."""
        return _coconut_tail_call((vector), *map(_coconut_minus, self.pts))
    @_coconut_tco
    def __mul__(self, other):
        """Scalar multiplication and dot product."""
        _coconut_match_to = other
        _coconut_match_check = False
        if _coconut.isinstance(_coconut_match_to, vector):
            other_pts = _coconut_match_to[0:]
            _coconut_match_check = True
        if _coconut_match_check:
            assert len(other_pts) == len(self.pts)
            return _coconut_tail_call((sum), map(_coconut.operator.mul, self.pts, other_pts))  # dot product
        else:
            return _coconut_tail_call((vector), *map(_coconut.functools.partial(_coconut.operator.mul, other), self.pts))  # scalar multiplication
    def __rmul__(self, other):
        """Necessary to make scalar multiplication commutative."""
        return self * other

"""
    Turn all the tuples in linearized_plane into vectors,
    using the n-vector class we defined earlier.
"""
@_coconut_tco
def vector_field():
    return _coconut_tail_call(starmap, vector, linearized_plane())

# You'll need to bring in the vector class from earlier to make these work
(print)(_coconut_igetitem(vector_field(), 0))  # vector(*pts=(0, 0))
(print)((list)(_coconut_igetitem(vector_field(), _coconut.slice(2, 3))))  # [vector(*pts=(1, 0))]
(print)((list)(_coconut_igetitem(vector_field(), _coconut.slice(1, 3))))

(print)((list)(map(abs, _coconut_igetitem(vector_field(), _coconut.slice(None, 5)))))
