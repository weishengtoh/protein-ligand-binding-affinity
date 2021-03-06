from abc import ABC, abstractmethod
from typing import Dict, Any


class History(ABC):

    def __init__(self, model_name: str, config):
        self.model_name = model_name
        self.config = config

    def get_name(self) -> str:
        return self.model_name

    def get_name(self):
        return self.config

    def set_params(self, params: Dict[str, Any]) -> None:
        self.params = params

    def get_params(self) -> Dict[str, Any]:
        return self.params