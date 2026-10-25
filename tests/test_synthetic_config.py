"""Synthetic option validation, corpus overrides, and effective report settings."""

from __future__ import annotations

import json
from itertools import islice

import pytest
import yaml

from ltm100 import cli
from ltm100.adapters.datasets.synthetic import SyntheticAdapter
from ltm100.config import AdapterConfig, dataset_settings
from ltm100.core.memory_growth import NamespacedDataset
from ltm100.core.scenarios import Mixed, SearchLoad


@pytest.mark.parametrize(
    "field,values",
    [
        ("memories_per_user", [0, -1, 2.5, 2.0, True, False, "2", None]),
        ("content_chars", [0, -1, 200.5, 200.0, True, False, "200", None]),
        ("categories", [-1, 2.5, 2.0, True, False, "2", None]),
        ("corpus_seed", [1.5, 1.0, True, False, "42"]),
    ],
)
def test_invalid_options_fail_at_construction(field, values):
    for value in values:
        with pytest.raises(ValueError, match=rf"synthetic\.{field} must be"):
            SyntheticAdapter(**{field: value})


@pytest.mark.parametrize("corpus_seed", [None, 0, 42, -7, 2**80 + 3])
def test_override_preserves_legacy_corpus_and_user_identities(corpus_seed):
    snapshots = []
    for run_seed in (0, 19, -23):
        dataset = SyntheticAdapter(
            memories_per_user=6,
            content_chars=80,
            categories=3,
            corpus_seed=corpus_seed,
        )
        reference = SyntheticAdapter(
            memories_per_user=6, content_chars=80, categories=3
        )
        users = dataset.users(3, seed=run_seed)
        assert users == reference.users(
            3, seed=run_seed if corpus_seed is None else corpus_seed
        )
        corpus = {user: list(dataset.memory_stream(user)) for user in users}
        assert corpus == {user: list(reference.memory_stream(user)) for user in users}
        snapshots.append(corpus)
    if corpus_seed is not None:
        assert snapshots[0] == snapshots[1] == snapshots[2]
    else:
        assert snapshots[0] != snapshots[1] != snapshots[2]


def test_override_applies_before_users_and_to_new_streams_after_reseeding():
    dataset = SyntheticAdapter(memories_per_user=4, corpus_seed=-7)
    user = "syn_user_00000"
    expected = list(dataset.memory_stream(user))
    dataset.users(1, seed=42)
    stream = dataset.memory_stream(user)
    first = next(stream)
    dataset.users(10, seed=99)
    assert [first, *stream] == expected
    assert list(dataset.memory_stream(user)) == expected


def test_fixed_corpus_does_not_freeze_workload_randomness():
    plans = []
    for seed in (7, 13):
        dataset = SyntheticAdapter(memories_per_user=20, corpus_seed=42)
        user = dataset.users(1, seed=seed)[0]
        scenario = Mixed(search_weight=0.5, think=1)
        operations = list(islice(scenario.plan(user, dataset, {"seed": seed}), 30))
        contents = {item.content for item in dataset.memory_stream(user)}
        for op in operations:
            if op.query is not None:
                assert op.query.query in contents
            else:
                assert all(item.content in contents for item in op.items)
        plans.append([(op.type, op.delay) for op in operations])
    assert plans[0] != plans[1]


def test_fixed_corpus_query_prefix_survives_growth_and_namespaces():
    expected = SyntheticAdapter(memories_per_user=4, corpus_seed=42)
    source_users = expected.users(2, seed=0)
    contents = {u: [i.content for i in expected.memory_stream(u)] for u in source_users}
    for seed, count, namespace in ((0, 4, "first_"), (17, 9, "second_")):
        dataset = NamespacedDataset(
            SyntheticAdapter(memories_per_user=count, corpus_seed=42), namespace
        )
        for user, source in zip(dataset.users(2, seed=seed), source_users):
            items = list(islice(dataset.memory_stream(user), 4))
            assert [item.content for item in items] == contents[source]
            assert {item.producer for item in items} == {user}
            scenario = SearchLoad(query_limit=4)
            ops = list(islice(scenario.plan(user, dataset, {"seed": seed}), 4))
            assert {op.query.query for op in ops} == set(contents[source])


def test_size_semantics_and_legacy_undershoot_remain_unchanged():
    dataset = SyntheticAdapter()
    user = dataset.users(1)[0]
    item = next(dataset.memory_stream(user))
    assert len(item.content) == len(item.content.encode("utf-8")) == 225
    assert len(item.content.removeprefix(f"{user} memory 0: ")) == 200
    dataset = SyntheticAdapter(memories_per_user=243, content_chars=8)
    user = dataset.users(1)[0]
    item = next(islice(dataset.memory_stream(user), 242, None))
    assert item.content == f"{user} memory 242: a dog a"


@pytest.mark.parametrize(
    "override,source", [(None, "run_seed"), (0, "dataset"), (-7, "dataset")]
)
def test_settings_include_effective_defaults_without_generating_data(
    monkeypatch, override, source
):
    def unexpected(*args, **kwargs):
        raise AssertionError("metadata must not enumerate users or generate memories")

    monkeypatch.setattr(SyntheticAdapter, "users", unexpected)
    monkeypatch.setattr(SyntheticAdapter, "memory_stream", unexpected)
    options = {} if override is None else {"corpus_seed": override}
    settings = dataset_settings(AdapterConfig("synthetic", options), seed=42)
    assert settings == {
        "generator": "legacy-filler-v1",
        "memories_per_user": 100,
        "content_chars": 200,
        "content_chars_scope": "filler",
        "categories": 0,
        "corpus_seed": 42 if override is None else override,
        "corpus_seed_source": source,
    }
    assert json.loads(json.dumps(settings)) == settings


@pytest.mark.parametrize("name", ["longmemeval", "nebius"])
def test_settings_do_not_construct_other_datasets(monkeypatch, name):
    def unexpected(*args, **kwargs):
        raise AssertionError("metadata must not construct other datasets")

    monkeypatch.setattr("ltm100.config.build_dataset", unexpected)
    assert (
        dataset_settings(AdapterConfig(name, {"path": "absent.json"}), seed=0) is None
    )


def _config(tmp_path, options):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"name": "synthetic", **options},
                "backend": {"name": "memmachine"},
            }
        )
    )
    return str(path)


@pytest.mark.parametrize("procs", [1, 2])
@pytest.mark.parametrize(
    "options",
    [
        {"memories_per_user": 0},
        {"content_chars": 0},
        {"categories": 2.5},
        {"corpus_seed": True},
    ],
)
def test_cli_rejects_invalid_options_before_backend_or_workers(
    tmp_path, monkeypatch, procs, options
):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid dataset must fail before probes or workers")

    for name in (
        "_backend_build",
        "_probe_server_metrics",
        "run_shards",
        "build_backend",
    ):
        monkeypatch.setattr(cli, name, unexpected)
    args = cli.build_parser().parse_args(
        [
            "run",
            "--config",
            _config(tmp_path, options),
            "--scenario",
            "add-load",
            "--users",
            "2",
            "--procs",
            str(procs),
            "--ops",
            "4",
            "--server-metrics",
        ]
    )
    with pytest.raises(ValueError, match="synthetic\\."):
        cli._run(args)


@pytest.mark.parametrize("override", [None, 0, 42])
@pytest.mark.parametrize("procs", [1, 2])
def test_cli_reports_effective_settings_in_stdout_and_json(
    tmp_path, monkeypatch, capsys, override, procs
):
    options = {
        "memories_per_user": 12,
        "content_chars": 80,
        "categories": 3,
        "corpus_seed": override,
    }
    monkeypatch.setattr(cli, "_backend_build", lambda cfg: {})
    monkeypatch.setattr(cli, "run_shards", lambda *args: [])
    output = tmp_path / "out"
    args = cli.build_parser().parse_args(
        [
            "run",
            "--config",
            _config(tmp_path, options),
            "--scenario",
            "add-load",
            "--users",
            "2",
            "--procs",
            str(procs),
            "--seed",
            "19",
            "--ops",
            "4",
            "--output",
            str(output),
        ]
    )
    assert cli._run(args) == 0
    payload = json.loads((output / "summary.json").read_text())
    printed = capsys.readouterr().out.split("\nreports written to", 1)[0]
    assert json.loads(printed) == payload
    assert payload["meta"]["dataset"] == "synthetic"
    assert payload["meta"]["seed"] == 19
    assert payload["meta"]["ops"] == 4
    assert payload["meta"]["dataset_config"] == dataset_settings(
        AdapterConfig("synthetic", options), seed=19
    )


def test_invalid_sweep_fails_without_backend_or_output(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid sweep must not reach a backend run")

    monkeypatch.setattr(cli, "_run", unexpected)
    output = tmp_path / "sweep"
    args = cli.build_parser().parse_args(
        [
            "sweep",
            "memory-growth",
            "--config",
            _config(tmp_path, {"categories": -1}),
            "--memory-counts",
            "2,4",
            "--queries-per-user",
            "2",
            "--ops",
            "4",
            "--output",
            str(output),
        ]
    )
    with pytest.raises(ValueError, match="synthetic.categories"):
        cli._run_memory_growth(args)
    assert not output.exists()
