"""Ingestion pipeline package."""
from .DataProcessor import DataProcessor, DefaultDataProcessor
from .schema import JSONDict
__all__ = [
    "JSONDict",
    "DataProcessor",
    "DefaultDataProcessor",
]