"""tools/filter_contract.py: who owns the fixture project, and the precedence canary.

The script's HTTP helpers are replaced, so none of this needs a server.
"""

from __future__ import annotations

import importlib.util
import io
import pathlib
import urllib.error

_PATH = pathlib.Path(__file__).resolve().parent.parent / "tools" / "filter_contract.py"
_spec = importlib.util.spec_from_file_location("filter_contract", _PATH)
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)

CANARY = "an unparenthesised OR is still the hazard"


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://core:8081", code, "error", {}, io.BytesIO(b""))


class _Server:
    """Stands in for post(): answers by path and records every call."""

    def __init__(self, *, exists=False, lookup_code=None, create_code=201, fail_load=False):
        self.exists = exists
        self.lookup_code = lookup_code
        self.create_code = create_code
        self.fail_load = fail_load
        self.calls: list[str] = []

    def __call__(self, base, path, body):
        self.calls.append(path)
        if path == "/api/v2/projects/get":
            if self.lookup_code:
                raise _http(self.lookup_code)
            if not self.exists:
                raise _http(404)
        elif path == "/api/v2/projects" and self.create_code >= 400:
            raise _http(self.create_code)
        elif path == "/api/v2/memories" and self.fail_load:
            raise _http(500)
        elif path == "/api/v2/memories/search":
            return {"content": {"episodic_memory": {"long_term_memory": {"episodes": []}}}}
        return {}

    @property
    def deleted(self) -> bool:
        return "/api/v2/projects/delete" in self.calls


def _main(monkeypatch, server, *args):
    monkeypatch.setattr(fc, "post", server)
    return fc.main(["filter_contract.py", "http://core:8081", *args])


def test_a_project_that_already_existed_is_never_deleted(monkeypatch):
    s = _Server(exists=True)
    _main(monkeypatch, s)
    assert "/api/v2/projects" not in s.calls, "created a project that already existed"
    assert not s.deleted


def test_a_project_this_run_created_is_deleted(monkeypatch):
    s = _Server()
    _main(monkeypatch, s)
    assert s.deleted


def test_a_half_built_fixture_this_run_created_is_still_removed(monkeypatch):
    s = _Server(fail_load=True)
    assert _main(monkeypatch, s) == 2
    assert s.deleted


def test_a_create_answered_409_is_not_ours(monkeypatch):
    """Absent at the lookup, then a 409: it exists with a different config, so
    someone else made it in between."""
    s = _Server(create_code=409)
    _main(monkeypatch, s)
    assert not s.deleted


def test_a_failed_lookup_deletes_nothing(monkeypatch):
    """The old probe swallowed this error and read it as an empty project."""
    s = _Server(lookup_code=500)
    assert _main(monkeypatch, s) == 2
    assert not s.deleted


def test_keep_leaves_even_a_project_this_run_created(monkeypatch):
    s = _Server()
    _main(monkeypatch, s, "--keep")
    assert not s.deleted


def _canary(monkeypatch, episodes):
    monkeypatch.setattr(fc, "search", lambda *a, **k: episodes)
    _, failure, note = next(r for r in fc.checks("http://core:8081", "o") if r[0].startswith(CANARY))
    return failure, note


def _ep(producer):
    return {"producer_id": producer, "metadata": {"category": "cat_1"}}


def test_the_canary_fails_on_an_empty_result(monkeypatch):
    failure, _ = _canary(monkeypatch, [])
    assert failure, "an empty result says nothing about precedence, so it cannot pass"


def test_a_precedence_change_is_a_note_not_a_failure(monkeypatch):
    failure, note = _canary(monkeypatch, [_ep("cu_00")])
    assert not failure
    assert "precedence" in note


def test_the_hazard_still_present_passes_quietly(monkeypatch):
    assert _canary(monkeypatch, [_ep("cu_00"), _ep("cu_01")]) == ("", "")
