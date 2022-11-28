from collections.abc import Callable
from pipelines import Pipeline, PipelineFunction

registry: dict[str, Pipeline] = {}


def register(name: str) -> Callable[[PipelineFunction], Pipeline]:
    def decorator(pipeline: PipelineFunction):
        registry[name] = Pipeline(name, pipeline)
        return pipeline

    return decorator
