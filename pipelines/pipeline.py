from collections.abc import Callable
from jinja2 import Environment


PipelineFunction = Callable[[list[list[str]], Environment], None]


class Pipeline:
    def __init__(self, name: str, func: PipelineFunction):
        self.name = name
        self.func = func

    def __call__(self, cards: list[list[str]], env: Environment):
        self.func(cards, env)
