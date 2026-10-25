"""Run-seed propagation through workloads, pre-ingest, and process sharding."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ltm100.adapters.datasets.synthetic import SyntheticAdapter
from ltm100.common import ResultItem
from ltm100.core.config import RunConfig
from ltm100.core.multiproc import run_shards
from ltm100.core.runner import LoadRunner
from ltm100.core.scenarios import AddLoad, Mixed, SearchLoad


def _seeded_entry(args: dict, proc_index: int):
    class CapturedDataset(SyntheticAdapter):
        def __init__(self):
            super().__init__(
                memories_per_user=6,
                content_chars=80,
                categories=3,
                corpus_seed=args["corpus_seed"],
            )
            self.corpus = {}

        def memory_stream(self, user):
            corpus = self.corpus.setdefault(user, {})
            for index, item in enumerate(super().memory_stream(user)):
                corpus[index] = asdict(item)
                yield item

    class CapturedBackend:
        name = "seed-capture"

        def __init__(self):
            self.adds = {}
            self.searches = {}

        async def setup(self, users):
            pass

        async def add(self, user, items):
            self.adds.setdefault(user, []).extend(item.content for item in items)
            await asyncio.sleep(0)
            return ["id"] * len(items)

        async def search(self, user, query):
            self.searches.setdefault(user, []).append(query.query)
            await asyncio.sleep(0)
            return [ResultItem(content="hit")]

    async def run():
        dataset = CapturedDataset()
        backend = CapturedBackend()
        scenario = {
            "add-load": AddLoad(),
            "search-load": SearchLoad(query_limit=3),
            "mixed": Mixed(think=0),
        }[args["scenario"]]
        runner = LoadRunner(
            client=backend,
            dataset=dataset,
            scenario=scenario,
            config=RunConfig(
                users=4,
                seed=args["seed"],
                ops=64 // args["procs"],
                procs=args["procs"],
                proc_index=proc_index,
                global_concurrency=2,
                preingest=args["scenario"] != "add-load",
                preingest_items_per_user=5 if args["scenario"] != "add-load" else None,
            ),
        )
        results = await runner.run()
        assert len(results) == 64 // args["procs"]
        assert all(result.status == "ok" for result in results)
        Path(args["traces"], f"{proc_index}.json").write_text(
            json.dumps(
                {
                    "corpus": {
                        user: list(items.values())
                        for user, items in dataset.corpus.items()
                    },
                    "adds": backend.adds,
                    "searches": backend.searches,
                }
            )
        )
        return results

    return asyncio.run(run())


@pytest.mark.parametrize("scenario", ["add-load", "search-load", "mixed"])
@pytest.mark.parametrize(
    "seed,corpus_seed", [(0, None), (42, None), (-7, None), (42, 0), (19, 42), (-7, 42)]
)
def test_workloads_preserve_seeded_corpus_across_process_counts(
    tmp_path, scenario, seed, corpus_seed
):
    reference = SyntheticAdapter(
        memories_per_user=6,
        content_chars=80,
        categories=3,
        corpus_seed=corpus_seed,
    )
    expected = {
        user: list(reference.memory_stream(user))
        for user in reference.users(4, seed=seed)
    }
    snapshots = []
    for procs in (1, 2):
        traces = tmp_path / str(procs)
        traces.mkdir()
        results = run_shards(
            _seeded_entry,
            {
                "scenario": scenario,
                "seed": seed,
                "corpus_seed": corpus_seed,
                "procs": procs,
                "traces": str(traces),
            },
            procs,
        )
        assert len(results) == 64
        corpus = {}
        for path in traces.glob("*.json"):
            trace = json.loads(path.read_text())
            corpus.update(trace["corpus"])
            for user, items in trace["corpus"].items():
                assert items == [asdict(item) for item in expected[user][: len(items)]]
            for user, contents in trace["adds"].items():
                assert set(contents) <= {item.content for item in expected[user]}
            for user, queries in trace["searches"].items():
                pool = (
                    expected[user][:3] if scenario == "search-load" else expected[user]
                )
                assert set(queries) <= {item.content for item in pool}
        assert set(corpus) == set(expected)
        snapshots.append(corpus)
    assert snapshots[0] == snapshots[1]
