"""Extraction agent (T9 baseline 3): a tool-calling loop that replaces the
generate -> review -> revise steps for one chunk at a time.

- library.py  skills (SKILL.md folders) and lessons, read-only during runs
- memory.py   act memory: what the agent learned so far about the current act
- tools.py    the tools the model can call (all offline)
- loop.py     the episode loop per chunk and run_act() over an act's chunks
"""
from .loop import AgentConfig, ChunkResult, run_act, run_chunk

__all__ = ["AgentConfig", "ChunkResult", "run_act", "run_chunk"]
