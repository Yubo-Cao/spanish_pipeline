from collections.abc import Callable
from jinja2 import Environment
from typing import Any

PipelineFunction = Callable[[list[list[str]], dict[str, Any]], None]


class Pipeline:
    def __init__(self, name: str, func: PipelineFunction):
        self.name = name
        self.func = func

    def __call__(self, cards: list[list[str]], env: dict[str, Any]):
        self.func(cards, env)
