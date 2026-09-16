"""Stable reader-facing imports for the MUSE-IDR model implementations.

The canonical research implementations remain under ``experiments`` so that
the code used by each recorded experiment is preserved verbatim. Import the
specific compatibility module (``models.a2``, ``models.a8``, or ``models.a10``)
to avoid loading optional deep-learning dependencies unnecessarily.
"""
