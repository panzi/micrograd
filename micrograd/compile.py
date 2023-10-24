from typing import Any, Callable, Optional

from .nn import MLP
from .engine import Node, Value, Add, Mul, Pow, ReLU

def compile(model: MLP, scores: list[Node], total_loss: Node, inputs: Optional[list[Node]]=None) -> tuple[Callable[[float], float], Callable[[], list[float]], Callable[[], list[tuple[float, float]]], Callable[[list[float]], None]]:
    code: list[str] = [
        'from typing import Callable\n'
        '\n'
        'def construct() -> tuple[Callable[[float], float], Callable[[], list[float]], Callable[[], list[tuple[float, float]]], Callable[[list[float]], None]]:\n'
    ]

    value_map: dict[Node, int] = {}

    def get_index(node: Node) -> int:
        index = value_map.get(node)
        if index is None:
            value_map[node] = index = len(value_map)
        return index

    def forward(node: Node):
        if node not in visited:
            visited.add(node)

            index = get_index(node)

            if isinstance(node, Value):
                code.append(f'        g{index} = 0.0\n')

            elif isinstance(node, Add):
                lhs = node._lhs
                rhs = node._rhs
                lhs_index = get_index(lhs)
                rhs_index = get_index(rhs)

                forward(lhs)
                forward(rhs)

                code.append(f'        v{index} = v{lhs_index} + v{rhs_index}\n')
                code.append(f'        g{index} = 0.0\n')

            elif isinstance(node, Mul):
                lhs = node._lhs
                rhs = node._rhs
                lhs_index = get_index(lhs)
                rhs_index = get_index(rhs)

                forward(lhs)
                forward(rhs)

                code.append(f'        v{index} = v{lhs_index} * v{rhs_index}\n')
                code.append(f'        g{index} = 0.0\n')

            elif isinstance(node, Pow):
                lhs = node._lhs
                rhs = node._rhs
                lhs_index = get_index(lhs)

                forward(lhs)

                code.append(f'        v{index} = v{lhs_index} ** {rhs!r}\n')
                code.append(f'        g{index} = 0.0\n')

            elif isinstance(node, ReLU):
                arg = node._arg
                arg_index = get_index(arg)

                forward(arg)

                code.append(f'        v{index} = 0.0 if v{arg_index} < 0.0 else v{arg_index}\n')
                code.append(f'        g{index} = 0.0\n')

            else:
                raise TypeError(f'unhandeled type of node {node!r}')

    def backward(node: Node):
        topo: list[Node] = []
        node._build_topo(visited, topo)

        index = get_index(total_loss)

        code.append(
            f'        g{index} = 1.0\n'
        )

        topo.reverse()

        for node in topo:
            index = get_index(node)

            if isinstance(node, Value):
                pass

            elif isinstance(node, Add):
                lhs = node._lhs
                rhs = node._rhs
                lhs_index = get_index(lhs)
                rhs_index = get_index(rhs)

                code.append(f'        g{lhs_index} += g{index}\n')
                code.append(f'        g{rhs_index} += g{index}\n')

            elif isinstance(node, Mul):
                lhs = node._lhs
                rhs = node._rhs
                lhs_index = get_index(lhs)
                rhs_index = get_index(rhs)

                code.append(f'        g{lhs_index} += v{rhs_index} * g{index}\n')
                code.append(f'        g{rhs_index} += v{lhs_index} * g{index}\n')

            elif isinstance(node, Pow):
                lhs = node._lhs
                rhs = node._rhs
                lhs_index = get_index(lhs)

                code.append(f'        g{lhs_index} += ({rhs!r} * v{lhs_index} ** {(rhs - 1)!r}) * g{index}\n')

            elif isinstance(node, ReLU):
                arg = node._arg
                arg_index = get_index(arg)

                code.append(f'        g{arg_index} += (v{index} > 0.0) * g{index}\n')

            else:
                raise TypeError(f'unhandeled type of node {node!r}')

    visited: set[Node] = set()

    parameters = list(model.parameters())

    for param in parameters:
        if param not in value_map:
            value_map[param] = len(value_map)

    for score in scores:
        if score not in value_map:
            value_map[score] = len(value_map)

    if total_loss not in value_map:
        value_map[total_loss] = len(value_map)

    init_index = len(code)
    code.append('')

    code.append(
        '    def eval_model(learning_rate: float) -> float:\n'
    )

    nonlocal_index = len(code)
    code.append('        nonlocal')

    forward(total_loss)
    visited.clear()

    backward(total_loss)

    init = ''.join(
        f'    v{index} = {value.data!r}\n'
        f'    g{index} = 0.0\n'

        for value, index in value_map.items()
    )
    code[init_index] = init

    nonlocals_str = ', '.join(f'v{index}, g{index}' for index in sorted(value_map.values()))
    code[nonlocal_index] = f'        nonlocal {nonlocals_str}\n'

    for param in parameters:
        index = value_map[param]
        code.append(f'        v{index} -= learning_rate * g{index}\n')

    code.append(
        f'        return v{value_map[total_loss]}\n'
    )

    code.append(
        '\n'
        '    def get_scores() -> list[float]:\n'
        '        return ['
    )

    code.append(', '.join(f'v{value_map[score]}' for score in scores))
    code.append(']\n')

    code.append(
        '\n'
        '    def get_parameters() -> list[tuple[float, float]]:\n'
        '        return ['
    )

    code.append(', '.join(f'(v{value_map[param]}, g{value_map[param]})' for param in parameters))
    code.append(']\n')

    code.append(
        '\n'
        '    def set_inputs(inputs: list[float]) -> None:\n'
    )

    if inputs:
        nonlocals: list[str] = []
        vals: list[str] = []
        grads: list[str] = []
        for input in inputs:
            local_index = value_map.get(input)
            if local_index is not None:
                v = f'v{local_index}'
                g = f'g{local_index}'
                vals.append(v)
                grads.append(g)
                nonlocals.append(v)
                nonlocals.append(g)
            else:
                vals.append('_')

        if nonlocals:
            code.append(f'        nonlocal {", ".join(nonlocals)}\n')
        code.append(f'        {", ".join(vals)}, = inputs\n')
        if grads:
            code.append(f'        {" = ".join(grads)} = 0.0\n')
    else:
        code.append('        pass\n')

    code.append(
        '\n'
        '    return (eval_model, get_scores, get_parameters, set_inputs)\n'
    )

    global_vars: dict[str, Any] = {}

    source = ''.join(code)

    # DEBUG:
    #with open("/tmp/compiled.py", "w") as fp:
    #    fp.write(source)

    exec(source, {}, global_vars)
    return global_vars['construct']()
