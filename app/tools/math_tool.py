"""Math tool with AST-based safe expression evaluation.

Provides accurate arithmetic calculations to prevent LLM computational errors.
Supports arithmetic operations, math functions, trigonometric functions,
rounding operations, and mathematical constants.
"""

import ast
import math
import operator
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from app.tools.registry import get_tool_registry


class MathToolInput(BaseModel):
    """Input schema for MathTool."""

    expression: str = Field(
        description="Mathematical expression to evaluate. "
        "Supports: +, -, *, /, //, %, ** (arithmetic), "
        "sqrt, pow, exp, log, log10, log2 (math functions), "
        "sin, cos, tan, asin, acos, atan (trigonometry), "
        "round, ceil, floor, trunc (rounding), "
        "abs, min, max, sum, factorial, gcd (utilities), "
        "pi, e, tau (constants). "
        "Examples: '2 + 3 * 4', 'sqrt(16)', 'sin(pi/2)', 'round(3.14159, 2)'"
    )
    decimal_places: int | None = Field(
        default=None,
        description="Number of decimal places for the result. If None, returns full precision."
    )


# Supported binary operators
_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Supported unary operators
_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Supported functions
_FUNCTIONS: dict[str, Any] = {
    # Math functions
    "sqrt": math.sqrt,
    "pow": pow,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    # Trigonometric functions
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    # Rounding functions
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "trunc": math.trunc,
    # Utility functions
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "factorial": math.factorial,
    "gcd": math.gcd,
}

# Supported constants
_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

# Full-width to half-width character mapping
_FULLWIDTH_MAP: dict[str, str] = {
    "＋": "+",
    "－": "-",
    "×": "*",
    "÷": "/",
    "（": "(",
    "）": ")",
    "．": ".",
    "０": "0",
    "１": "1",
    "２": "2",
    "３": "3",
    "４": "4",
    "５": "5",
    "６": "6",
    "７": "7",
    "８": "8",
    "９": "9",
}


def _preprocess_expression(expression: str) -> str:
    """Preprocess expression for evaluation.

    - Remove commas (thousand separators)
    - Convert full-width characters to half-width

    Args:
        expression: Raw expression string

    Returns:
        Preprocessed expression
    """
    result = expression

    # Remove commas (thousand separators like 1,000,000)
    result = result.replace(",", "")

    # Convert full-width characters
    for fullwidth, halfwidth in _FULLWIDTH_MAP.items():
        result = result.replace(fullwidth, halfwidth)

    return result.strip()


def _safe_eval_node(node: ast.AST) -> Any:
    """Safely evaluate an AST node.

    Args:
        node: AST node to evaluate

    Returns:
        Evaluated result

    Raises:
        ValueError: If node type is not supported (security)
    """
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    # Numbers (int, float)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    # Named constants (pi, e, tau)
    if isinstance(node, ast.Name):
        name = node.id.lower()
        if name in _CONSTANTS:
            return _CONSTANTS[name]
        raise ValueError(f"Unknown constant: {node.id}")

    # Unary operations (-x, +x)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval_node(node.operand)
        return _UNARY_OPS[op_type](operand)

    # Binary operations (x + y, x * y, etc.)
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _BINARY_OPS[op_type](left, right)

    # Function calls (sqrt(x), sin(x), etc.)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are supported")
        func_name = node.func.id.lower()
        if func_name not in _FUNCTIONS:
            raise ValueError(f"Unknown function: {node.func.id}")
        args = [_safe_eval_node(arg) for arg in node.args]
        return _FUNCTIONS[func_name](*args)

    # List (for min, max, sum)
    if isinstance(node, ast.List):
        return [_safe_eval_node(elem) for elem in node.elts]

    # Tuple (for min, max, sum)
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_node(elem) for elem in node.elts)

    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def evaluate_expression(expression: str, decimal_places: int | None = None) -> str:
    """Safely evaluate a mathematical expression using AST parsing.

    Args:
        expression: Mathematical expression string
        decimal_places: Optional number of decimal places for result

    Returns:
        String representation of the result

    Raises:
        ValueError: If expression is invalid or contains unsupported operations
    """
    # Preprocess expression
    processed = _preprocess_expression(expression)

    if not processed:
        raise ValueError("Empty expression")

    logger.debug(f"Evaluating expression: {processed}")

    # Parse to AST
    try:
        tree = ast.parse(processed, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")

    # Evaluate safely
    result = _safe_eval_node(tree)

    # Apply decimal places if specified
    if decimal_places is not None and isinstance(result, (int, float)):
        result = round(result, decimal_places)

    logger.debug(f"Expression result: {result}")

    return str(result)


async def math_calculate(expression: str, decimal_places: int | None = None) -> str:
    """Calculate a mathematical expression safely.

    Uses AST-based parsing to prevent code injection vulnerabilities.
    Supports arithmetic operations, math functions, trigonometric functions,
    rounding operations, and mathematical constants.

    Args:
        expression: Mathematical expression to evaluate
        decimal_places: Optional number of decimal places for result

    Returns:
        Calculation result as string
    """
    try:
        result = evaluate_expression(expression, decimal_places)
        return f"Result: {result}"
    except ValueError as e:
        return f"Error: {e}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except OverflowError:
        return "Error: Result too large"
    except Exception as e:
        logger.error(f"Unexpected error in math calculation: {e}")
        return f"Error: {e}"


def register_math_tool() -> None:
    """Register the math tool with the global registry."""
    registry = get_tool_registry()

    # Check if already registered
    if registry.get("math") is not None:
        logger.debug("Math tool already registered")
        return

    @registry.register(
        name="math",
        description=(
            "Calculate mathematical expressions accurately. "
            "Use this tool for any arithmetic or mathematical calculations "
            "to ensure precision and avoid computational errors. "
            "Supports: arithmetic (+, -, *, /, //, %, **), "
            "math functions (sqrt, pow, exp, log, log10, log2), "
            "trigonometry (sin, cos, tan, asin, acos, atan), "
            "rounding (round, ceil, floor, trunc), "
            "utilities (abs, min, max, sum, factorial, gcd), "
            "constants (pi, e, tau). "
            "Examples: 'sqrt(16) + 2**3', 'sin(pi/2)', 'round(22/7, 4)'"
        ),
        category="math",
        args_schema=MathToolInput,
    )
    async def math(expression: str, decimal_places: int | None = None) -> str:
        """Calculate a mathematical expression."""
        return await math_calculate(expression, decimal_places)

    logger.debug("Math tool registered")


# Auto-register on module import for mandatory binding
register_math_tool()
