"""LLM baseline — serialize patient event sequences as text, query a locally-served LLM for event-occurrence
probabilities, and write ``PredictionSchema``-conformant parquets.

A drop-in alternative to ``EQ_predict`` whose output feeds the existing ``EQ_evaluate`` pipeline unchanged.
See ``README.md`` in this directory for serving setup and usage.
"""
