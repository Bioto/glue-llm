from abc import ABC, abstractmethod
from source.models.agent import Agent

class Executor(ABC):
    @abstractmethod
    def execute(self) -> str:
        pass