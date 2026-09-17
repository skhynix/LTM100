#!/usr/bin/env python3
"""Assert what the server does with a filter, before you trust a filtered number.

The unit suite covers the filter string the client builds. This covers the other
half: what MemMachine does with it. Every filter defect found here so far was on
this side, while the client-side tests stayed green -- including an `OR` that
escaped a producer_id scope and returned every user's episodes.

Not a load arm. It ingests a small fixture, makes a dozen searches and asserts
semantics, so run it as a precondition whenever the build, the vector store or
the filter grammar changes.

    ./filter_contract.py http://localhost:8081
    ./filter_contract.py http://localhost:8081 --org contract2 --keep

It writes into its own org (default `filtercontract`) and deletes the project
afterwards if this run created it, unless --keep is passed. A project that
already existed is never deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

PROJECT = "shared"
USERS = ["cu_00", "cu_01", "cu_02"]
CATEGORIES = 4
PER_USER = 24  # 24 / 4 categories = 6 per (user, category); under a top_k of 20


class Fail(Exception):
    pass


class Note(Exception):
    """The check passed, but found something worth reporting."""


def post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
    # projects/delete answers 200 with an empty body.
    return json.loads(raw) if raw.strip() else {}


def search(base: str, org: str, query: str, *, filt: str | None = None, top_k: int = 20) -> list[dict]:
    body = {
        "org_id": org,
        "project_id": PROJECT,
        "query": query,
        "top_k": top_k,
        "types": ["episodic"],
    }
    if filt is not None:
        body["filter"] = filt
    resp = post(base, "/api/v2/memories/search", body)
    return resp["content"]["episodic_memory"]["long_term_memory"]["episodes"]


def search_status(base: str, org: str, filt: str) -> tuple[int, str]:
    """The HTTP status for a filter, for the cases that must be rejected."""
    try:
        search(base, org, "anything", filt=filt)
        return 200, ""
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def project_exists(base: str, org: str) -> bool:
    """True or False from projects/get; any other answer raises."""
    try:
        post(base, "/api/v2/projects/get", {"org_id": org, "project_id": PROJECT})
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def prepare(base: str, org: str) -> bool:
    """Make sure the fixture project exists. Returns True only if this run created it.

    Ownership is proven rather than inferred: the project must be absent before
    the create, and the create must succeed. Anything else -- it already
    existed, another client made it in between, or the lookup failed -- leaves
    it not ours, and main() will not delete it.
    """
    if project_exists(base, org):
        print(f"  warning: {org}/{PROJECT} already exists and will not be deleted "
              "afterwards. If it holds episodes, this run adds to them and the "
              "match-count check fails. Pass --org for a clean fixture.")
        return False
    try:
        post(base, "/api/v2/projects", {"org_id": org, "project_id": PROJECT,
                                        "description": "filter contract fixture"})
    except urllib.error.HTTPError as e:
        # Re-creating an identical project answers 201; 409 means it exists
        # with a different config, so it is someone else's.
        if e.code == 409:
            return False
        raise
    return True


def load(base: str, org: str) -> None:
    for user in USERS:
        messages = [
            {
                "content": f"{user} note {i}: memory systems store and retrieve context",
                "producer": user,
                "metadata": {"category": f"cat_{i % CATEGORIES}"},
            }
            for i in range(PER_USER)
        ]
        post(base, "/api/v2/memories", {"org_id": org, "project_id": PROJECT,
                                        "types": ["episodic"], "messages": messages})


def cats(eps: list[dict]) -> set[str]:
    return {(e.get("metadata") or {}).get("category") for e in eps}


def producers(eps: list[dict]) -> set[str]:
    return {e.get("producer_id") for e in eps}


def checks(base: str, org: str) -> list[tuple[str, str, str]]:
    """Return (name, failure, note) per check -- failure empty when it passed."""
    out: list[tuple[str, str, str]] = []

    def check(name):
        def wrap(fn):
            try:
                fn()
                out.append((name, "", ""))
            except Note as e:
                out.append((name, "", str(e)))
            except Fail as e:
                out.append((name, str(e), ""))
            except Exception as e:  # a broken check must not read as a pass
                out.append((name, f"{type(e).__name__}: {e}", ""))
            return fn
        return wrap

    q = "memory systems store"
    me = USERS[0]

    @check("a shared project really is shared")
    def _():
        eps = search(base, org, q)
        if len(producers(eps)) < 2:
            raise Fail(
                f"only {producers(eps)} in an unfiltered search, so the rest of "
                "these checks would pass even with filtering broken"
            )

    @check("a producer filter returns exactly one producer")
    def _():
        eps = search(base, org, q, filt=f"producer_id = '{me}'")
        if producers(eps) != {me}:
            raise Fail(f"expected only {me}, got {sorted(producers(eps))}")

    @check("a producer filter that matches nobody returns nothing")
    def _():
        eps = search(base, org, q, filt="producer_id = 'cu_absent'")
        if eps:
            raise Fail(f"expected 0 episodes, got {len(eps)}")

    @check("a metadata filter returns only that category")
    def _():
        eps = search(base, org, q, filt="m.category = 'cat_1'")
        if cats(eps) != {"cat_1"}:
            raise Fail(f"expected only cat_1, got {sorted(c for c in cats(eps) if c)}")

    @check("a restrictive filter returns the match count, not a padded top_k")
    def _():
        filt = f"producer_id = '{me}' AND (m.category = 'cat_1')"
        eps = search(base, org, q, filt=filt, top_k=20)
        want = PER_USER // CATEGORIES
        if len(eps) != want:
            raise Fail(
                f"expected {want} matching episodes, got {len(eps)} -- "
                "a padded top_k would return non-matching rows"
            )
        if producers(eps) != {me} or cats(eps) != {"cat_1"}:
            raise Fail(f"padding leaked in: {sorted(producers(eps))} {sorted(c for c in cats(eps) if c)}")

    @check("AND narrows to the intersection of both terms")
    def _():
        eps = search(base, org, q, filt=f"producer_id = '{me}' AND (m.category = 'cat_2')")
        if producers(eps) != {me} or cats(eps) != {"cat_2"}:
            raise Fail(f"got {sorted(producers(eps))} {sorted(c for c in cats(eps) if c)}")

    @check("an OR inside a producer scope cannot widen past that producer")
    def _():
        # The regression that shipped: AND binds tighter than OR, so an
        # unparenthesised tail matched every producer.
        filt = f"producer_id = '{me}' AND (m.category = 'cat_1' OR m.category = 'cat_2')"
        eps = search(base, org, q, filt=filt)
        if producers(eps) != {me}:
            raise Fail(f"OR escaped the producer scope: {sorted(producers(eps))}")
        if not cats(eps) <= {"cat_1", "cat_2"}:
            raise Fail(f"OR admitted other categories: {sorted(c for c in cats(eps) if c)}")

    @check("an unparenthesised OR is still the hazard the client guards against")
    def _():
        # Asserts the server's precedence, not the client's string. The backend
        # parenthesises either way, so a precedence change is news rather than
        # a failure: it is reported as a note and the gate stays green. An empty
        # result proves nothing about precedence, so that one does fail.
        filt = f"producer_id = '{me}' AND m.category = 'cat_1' OR m.category = 'cat_2'"
        eps = search(base, org, q, filt=filt)
        if not eps:
            raise Fail("no results, so this says nothing about precedence")
        if producers(eps) == {me}:
            raise Note(
                "an unparenthesised OR no longer widens the scope; the server's "
                "AND/OR precedence changed, so the backend's parenthesising could "
                "be revisited"
            )

    @check("an unknown filter field is rejected, not ignored")
    def _():
        code, body = search_status(base, org, "no_such_field = 'x'")
        if code == 200:
            raise Fail("accepted an unknown field: a typo would silently widen a filter")
        if code != 422:
            raise Fail(f"expected 422, got {code}: {body[:160]}")

    @check("a system field keeps its underscore-free name")
    def _():
        code, _body = search_status(base, org, f"_producer_id = '{me}'")
        if code == 200:
            raise Fail("_producer_id was accepted; the documented name is producer_id")

    @check("the filterable system fields are the ten we document")
    def _():
        _code, body = search_status(base, org, "no_such_field = 'x'")
        documented = {"content_type", "created_at", "episode_type", "episode_uid",
                      "produced_for_id", "producer_id", "producer_role",
                      "sequence_num", "session_key", "timestamp"}
        missing = sorted(f for f in documented if f"'{f}'" not in body)
        if missing:
            raise Fail(f"the server no longer lists {missing} as filterable")

    @check("metadata needs its prefix")
    def _():
        code, _body = search_status(base, org, "category = 'cat_1'")
        if code == 200:
            raise Fail("a bare metadata key was accepted; the documented form is m.<name>")

    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base_url")
    ap.add_argument("--org", default="filtercontract")
    ap.add_argument("--keep", action="store_true", help="leave the fixture project behind")
    args = ap.parse_args(argv[1:])

    print(f"  fixture: {len(USERS)} producers x {PER_USER} episodes, {CATEGORIES} categories, "
          f"one project ({args.org}/{PROJECT})")
    results: list[tuple[str, str, str]] = []
    setup_error = ""
    # Delete only a project this run created: --org can be pointed at a real
    # tenant, and the fixture name is generic enough to collide. `owned` starts
    # False, so a failure before the create succeeds deletes nothing; once it is
    # True, a fixture that fails halfway is still removed rather than left to
    # poison the next run.
    owned = False
    try:
        owned = prepare(args.base_url, args.org)
        load(args.base_url, args.org)
        results = checks(args.base_url, args.org)
    except Exception as e:
        setup_error = f"{type(e).__name__}: {e}"
    finally:
        if owned and not args.keep:
            try:
                post(args.base_url, "/api/v2/projects/delete",
                     {"org_id": args.org, "project_id": PROJECT})
            except Exception as e:
                print(f"  note: could not delete the fixture project: {e}")

    if setup_error:
        print(f"  could not run the checks: {setup_error}")
        return 2

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, failure, note in results:
        if failure:
            failed += 1
            print(f"  FAIL  {name:<{width}}  {failure}")
        elif note:
            print(f"  note  {name:<{width}}  {note}")
        else:
            print(f"  ok    {name}")
    print(f"\n  {len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
