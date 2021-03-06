from typing import Any
from abc import ABCMeta, abstractmethod
import pickle


class PipeLine:
    def __init__(self):
        self.pipeline = {}

    def add_function(self, function: Any, input_from_pre: bool = True, *args, **kwargs):
        self.pipeline[len(self.pipeline) + 1] = {
            "function": function,
            "args": args,
            "kwargs": kwargs,
            "input_from_pre": input_from_pre,
        }

    def execute(self, data: Any = None):
        x = data

        for item in self.pipeline.values():
            if item["input_from_pre"]:
                x = item["function"](x, *item["args"], **item["kwargs"])
            else:
                x = item["function"](*item["args"], **item["kwargs"])

        return x

    # load, saveは動かない
    def load_pipeline(self, filename: str):
        self.pipeline = pickle.load(open(filename, "rb"))

    def save_pipeline(self, filename: str = "pipeline.pkl"):
        pickle.dump(self.pipeline, open(filename, "wb"))
