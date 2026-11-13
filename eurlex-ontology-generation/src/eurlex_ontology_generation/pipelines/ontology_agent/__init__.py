"""
Pipeline 'ontology_agent': the extraction agent (T9 baseline 3) replaces
generate -> review -> revise for one batch. Run through
scripts/run_ontology_batches.py --mode agent.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]

__version__ = "0.1"
