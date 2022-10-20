import itertools
import tqdm
import yaml
from jinja2 import Environment, FileSystemLoader
from pipelines import *
from argparse import ArgumentParser

pipeline_table = {k: [v] for k, v in registry.items()}
pipeline_table["all"] = list(registry.values())


def parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--pipeline",
        "-p",
        action="append",
        choices=pipeline_table.keys(),
        help="Pipelines to run",
        default=["all"],
    )
    parser.add_argument(
        "data",
        type=str,
        nargs="?",
        default="cards.yml",
        help="The data file to use",
    )
    args = parser.parse_args()
    if len(args.pipeline) >= 2:
        args.pipeline = args.pipeline[1:]  # remove default all
    return args


def main():
    args = parse()

    with open(args.data, encoding="utf-8") as f:
        cards = yaml.load(f, Loader=yaml.FullLoader)

    env = Environment(loader=FileSystemLoader("./assets"))
    pipelines = set(itertools.chain(*[pipeline_table[p] for p in args.pipeline]))
    bar = tqdm.tqdm(pipelines, desc="Running pipelines")
    for pipeline in bar:
        bar.set_description(f"Running {pipeline.name}")
        pipeline(cards, env)


if __name__ == "__main__":
    main()
