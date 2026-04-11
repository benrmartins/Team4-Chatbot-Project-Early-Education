from ingestion_pipeline import *
from project_config import *

if __name__ == "__main__":
    processor = DefaultDataProcessor()
    processor.process()
