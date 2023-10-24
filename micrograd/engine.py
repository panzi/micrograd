from typing import Union, Sequence
from abc import ABC, abstractmethod

class Node(ABC):
    __slots__ = 'data', 'grad', '_op', '_k'

    data: float
    grad: float
    _op: str
    _k: int

    def __init__(self, data: float, op: str):
        self.data = data
        self.grad = 0.0
        # internal variables used for autograd graph construction
        self._op = op # the op that produced this node, for graphviz / debugging / etc
        self._k = 0

    @abstractmethod
    def refresh(self, k: int) -> float:
        ...

    @abstractmethod
    def _backward(self) -> None:
        ...

    def assign(self, value: float) -> None:
        self.data = value
        self.grad = 0.0
        self._k = 0

    def __add__(self, other: Union['Node', int, float]) -> 'Node':
        other = other if isinstance(other, Node) else Value(other)
        return Add(self, other)

    def __mul__(self, other: Union['Node', int, float]) -> 'Node':
        other = other if isinstance(other, Node) else Value(other)
        return Mul(self, other)

    def __pow__(self, other: Union[float, int]) -> 'Node':
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        return Pow(self, other)

    def relu(self) -> 'Node':
        return ReLU(self)

    def __neg__(self): # -self
        return self * -1.0

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1.0

    def __rtruediv__(self, other): # other / self
        return other * self**-1.0

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    @abstractmethod
    def _build_topo(self, visited: set['Node'], topo: list['Node']) -> None:
        ...

    def backward(self):
        # topological order all of the children in the graph
        topo: list['Node'] = []
        visited: set['Node'] = set()
        self._build_topo(visited, topo)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        topo.reverse()
        for v in topo:
            v._backward()

    @abstractmethod
    def children(self) -> Sequence['Node']:
        ...

class Value(Node):
    """ stores a single scalar value and its gradient """

    __slots__ = ()

    def __init__(self, data: float):
        super().__init__(data, '')

    def refresh(self, k: int) -> float:
        if self._k < k:
            self._k = k
            self.grad = 0.0
        return self.data

    def _backward(self) -> None:
        pass

    def _build_topo(self, visited: set['Node'], topo: list['Node']) -> None:
        if self not in visited:
            visited.add(self)
            topo.append(self)

    def children(self) -> Sequence['Node']:
        return ()

class Add(Node):
    __slots__ = '_lhs', '_rhs'

    _lhs: Node
    _rhs: Node

    def __init__(self, lhs: Node, rhs: Node) -> None:
        data = lhs.data + rhs.data
        super().__init__(data, '+')
        self._lhs = lhs
        self._rhs = rhs

    def refresh(self, k: int) -> float:
        if self._k >= k:
            return self.data
        lhs = self._lhs.refresh(k)
        rhs = self._rhs.refresh(k)
        self.data = value = lhs + rhs
        self._k = k
        self.grad = 0.0
        return value
    
    def _backward(self) -> None:
        grad = self.grad
        self._lhs.grad += grad
        self._rhs.grad += grad

    def _build_topo(self, visited: set['Node'], topo: list['Node']) -> None:
        if self not in visited:
            visited.add(self)
            self._rhs._build_topo(visited, topo)
            self._lhs._build_topo(visited, topo)
            topo.append(self)

    def children(self) -> Sequence['Node']:
        return (self._lhs, self._rhs)

class Mul(Node):
    __slots__ = '_lhs', '_rhs'

    _lhs: Node
    _rhs: Node

    def __init__(self, lhs: Node, rhs: Node) -> None:
        data = lhs.data * rhs.data
        super().__init__(data, '*')
        self._lhs = lhs
        self._rhs = rhs

    def refresh(self, k: int) -> float:
        if self._k >= k:
            return self.data
        lhs = self._lhs.refresh(k)
        rhs = self._rhs.refresh(k)
        self.data = value = lhs * rhs
        self._k = k
        self.grad = 0.0
        return value

    def _backward(self) -> None:
        grad = self.grad
        lhs = self._lhs
        rhs = self._rhs
        lhs.grad += rhs.data * grad
        rhs.grad += lhs.data * grad

    def _build_topo(self, visited: set['Node'], topo: list['Node']) -> None:
        if self not in visited:
            visited.add(self)
            self._rhs._build_topo(visited, topo)
            self._lhs._build_topo(visited, topo)
            topo.append(self)

    def children(self) -> Sequence['Node']:
        return (self._lhs, self._rhs)

class Pow(Node):
    __slots__ = '_lhs', '_rhs'

    _lhs: Node
    _rhs: float

    def __init__(self, lhs: Node, rhs: float) -> None:
        data = lhs.data ** rhs
        super().__init__(data, f'**{rhs}')
        self._lhs = lhs
        self._rhs = rhs

    def refresh(self, k: int) -> float:
        if self._k >= k:
            return self.data
        lhs = self._lhs.refresh(k)
        rhs = self._rhs
        self.data = value = lhs ** rhs
        self._k = k
        self.grad = 0.0
        return value
    
    def _backward(self) -> None:
        lhs = self._lhs
        rhs = self._rhs
        lhs.grad += (rhs * lhs.data ** (rhs - 1)) * self.grad

    def _build_topo(self, visited: set['Node'], topo: list['Node']) -> None:
        if self not in visited:
            visited.add(self)
            self._lhs._build_topo(visited, topo)
            topo.append(self)

    def children(self) -> Sequence['Node']:
        return (self._lhs,)

class ReLU(Node):
    __slots__ = '_arg'

    _arg: Node

    def __init__(self, arg: Node) -> None:
        value = arg.data
        value = 0.0 if value < 0.0 else value
        super().__init__(value, f'ReLU')
        self._arg = arg

    def refresh(self, k: int) -> float:
        if self._k >= k:
            return self.data
        arg = self._arg.refresh(k)
        self.data = value = 0.0 if arg < 0.0 else arg
        self._k = k
        self.grad = 0.0
        return value

    def _backward(self) -> None:
        arg = self._arg
        arg.grad += (self.data > 0.0) * self.grad

    def _build_topo(self, visited: set['Node'], topo: list['Node']) -> None:
        if self not in visited:
            visited.add(self)
            self._arg._build_topo(visited, topo)
            topo.append(self)

    def children(self) -> Sequence['Node']:
        return (self._arg,)
