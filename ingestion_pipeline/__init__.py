"""Ingestion pipeline package."""
from .DataProcessor import DataProcessor, DefaultDataProcessor

__all__ = [
    "DataProcessor",
    "DefaultDataProcessor",
]