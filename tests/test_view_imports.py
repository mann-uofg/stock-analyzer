"""Import every view module.

The suite exercises the analysis layer thoroughly and never imported the views,
so a name that does not exist in ``views.common`` - a helper renamed, or one
added to a view before it was added to common - reached the browser as a bare
ImportError with the message redacted. pyflakes catches undefined names inside
a module but not names imported across modules, which is the gap this closes.

These are deliberately shallow: importing is the whole test. Rendering needs a
live Streamlit session, and the point here is only that every module a page
depends on can actually be loaded.
"""

from __future__ import annotations

import importlib

import pytest

VIEW_MODULES = [
    "views.common",
    "views.theme",
    "views.research",
    "views.watchlist",
    "views.portfolio",
    "views.news",
    "views.earnings",
]


@pytest.mark.parametrize("name", VIEW_MODULES)
def test_module_imports(name):
    assert importlib.import_module(name) is not None


@pytest.mark.parametrize("name", [m for m in VIEW_MODULES
                                  if m not in ("views.common", "views.theme")])
def test_every_page_exposes_render(name):
    """st.Page needs a callable entry point on each page module."""
    module = importlib.import_module(name)
    assert callable(getattr(module, "render", None)), f"{name} has no render()"


def test_app_module_imports():
    """The whole import graph app.py builds, in one go."""
    importlib.import_module("views.earnings")
    importlib.import_module("views.news")
    importlib.import_module("views.portfolio")
    importlib.import_module("views.research")
    importlib.import_module("views.watchlist")
