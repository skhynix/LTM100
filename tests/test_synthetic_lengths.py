"""Opt-in length distributions preserve dataset and workload contracts."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from itertools import islice

import pytest
from test_synthetic_config import _config

from ltm100 import cli
from ltm100.adapters.datasets import synthetic
from ltm100.adapters.datasets.synthetic import SyntheticAdapter
from ltm100.config import AdapterConfig, dataset_settings
from ltm100.core.memory_growth import NamespacedDataset
from ltm100.core.scenarios import AddLoad, Mixed, SearchLoad

UNIFORM = {"type": "uniform", "min_chars": 1, "max_chars": 127}
WEIGHTED = {
    "type": "weighted",
    "choices": [
        {"chars": 8, "weight": 70},
        {"chars": 200, "weight": 25},
        {"chars": 1024, "weight": 5},
    ],
}
DISTRIBUTIONS = [
    pytest.param(UNIFORM, id="uniform"),
    pytest.param(WEIGHTED, id="weighted"),
]


def _dataset(distribution, *, memories=12, seed=None, categories=3):
    return SyntheticAdapter(
        memories_per_user=memories,
        content_chars_distribution=distribution,
        corpus_seed=seed,
        categories=categories,
    )


def _lengths(dataset, user):
    return [
        len(item.content.removeprefix(f"{user} memory {index}: "))
        for index, item in enumerate(dataset.memory_stream(user))
    ]


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("seed", [0, 42, -7, 2**80 + 3])
def test_distribution_preserves_reproducible_prefix_and_identities(distribution, seed):
    small = _dataset(distribution, memories=12)
    large = _dataset(distribution, memories=23)
    users = small.users(3, seed=seed)
    assert users == large.users(5, seed=seed)[:3]
    baseline = SyntheticAdapter(categories=3)
    assert users == baseline.users(3, seed=seed)
    for user in users:
        corpus = list(small.memory_stream(user))
        assert corpus == list(small.memory_stream(user))
        assert corpus == list(large.memory_stream(user))[:12]
        assert [item.metadata for item in corpus] == [
            {"category": f"cat_{index % 3}"} for index in range(12)
        ]
        assert {item.producer for item in corpus} == {user}
        assert all(item.role is None and item.timestamp is None for item in corpus)
    assert list(small.memory_stream(users[0])) != list(small.memory_stream(users[1]))


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("override", [None, 0, -7])
def test_distribution_honors_corpus_seed_and_active_stream_state(
    distribution, override
):
    dataset = _dataset(distribution, seed=override)
    user = "syn_user_00000"
    before = list(dataset.memory_stream(user))
    dataset.users(1, seed=42)
    expected = list(dataset.memory_stream(user))
    stream = dataset.memory_stream(user)
    first = next(stream)
    dataset.users(1, seed=19)
    assert [first, *stream] == expected
    if override is None:
        assert before != expected != list(dataset.memory_stream(user))
    else:
        assert before == expected == list(dataset.memory_stream(user))


def test_uniform_is_inclusive_and_exact_even_for_small_filler_sizes():
    dataset = _dataset(
        {"type": "uniform", "min_chars": 1, "max_chars": 8}, memories=512
    )
    user = dataset.users(1)[0]
    lengths = _lengths(dataset, user)
    assert set(lengths) == set(range(1, 9))
    assert abs(sum(lengths) / len(lengths) - 4.5) < 0.25
    for size in (1, 8, 200, 10001):
        fixed = _dataset(
            {"type": "uniform", "min_chars": size, "max_chars": size}, memories=20
        )
        assert _lengths(fixed, user) == [size] * 20


def test_weighted_lengths_match_seeded_probabilities_and_exact_sizes():
    dataset = _dataset(WEIGHTED, memories=10000, seed=42)
    user = dataset.users(1)[0]
    counts = Counter(_lengths(dataset, user))
    assert set(counts) == {8, 200, 1024}
    for size, probability in ((8, 0.7), (200, 0.25), (1024, 0.05)):
        assert abs(counts[size] / 10000 - probability) < 0.02
    single = _dataset(
        {"type": "weighted", "choices": [{"chars": 8, "weight": 0.5}]}, memories=300
    )
    assert _lengths(single, user) == [8] * 300


@pytest.mark.parametrize("weight", [5e-324, 1e-320])
def test_tiny_weights_preserve_equal_probabilities(weight):
    sequences = []
    for scale in (1.0, weight):
        distribution = {
            "type": "weighted",
            "choices": [{"chars": 1, "weight": scale}, {"chars": 2, "weight": scale}],
        }
        dataset = _dataset(distribution, memories=10000, seed=42)
        user = dataset.users(1)[0]
        sequences.append(_lengths(dataset, user))
    assert sequences[0] == sequences[1]


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_length_draws_are_independent_of_content_rng_and_categories(
    monkeypatch, distribution
):
    original = _dataset(distribution, memories=100, seed=42)
    user = original.users(1)[0]
    expected = _lengths(original, user)
    exact_filler = synthetic._exact_filler

    def extra_content_draws(rng, size):
        for _ in range(27):
            rng.random()
        return exact_filler(rng, size)

    monkeypatch.setattr(synthetic, "_exact_filler", extra_content_draws)
    changed = _dataset(distribution, memories=100, seed=42, categories=0)
    changed.users(1, seed=99)
    assert _lengths(changed, user) == expected


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_new_generator_stream_is_lazy_and_metadata_does_not_generate(
    monkeypatch, distribution
):
    def unexpected(*args):
        raise AssertionError("generation must remain lazy")

    dataset = _dataset(distribution, memories=10**12)
    monkeypatch.setattr(synthetic, "_exact_filler", unexpected)
    stream = dataset.memory_stream("syn_user_00000")
    settings = dataset.generation_settings(seed=42)
    assert settings["generator"] == "distributed-filler-v1"
    with pytest.raises(AssertionError, match="lazy"):
        next(stream)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_settings_and_config_mutation_do_not_change_corpus(distribution):
    supplied = json.loads(json.dumps(distribution))
    dataset = _dataset(supplied, seed=42)
    expected = list(dataset.memory_stream("syn_user_00000"))
    settings = dataset.generation_settings(seed=99)
    assert settings["content_chars"] is None
    assert settings["content_chars_scope"] == "filler"
    assert settings["content_chars_distribution"] == distribution
    assert settings["corpus_seed"] == 42
    assert settings["corpus_seed_source"] == "dataset"
    supplied.clear()
    settings["content_chars_distribution"].clear()
    assert (
        dataset.generation_settings(seed=0)["content_chars_distribution"]
        == distribution
    )
    assert list(dataset.memory_stream("syn_user_00000")) == expected
    assert (
        list(pickle.loads(pickle.dumps(dataset)).memory_stream("syn_user_00000"))
        == expected
    )


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_source_query_and_replay_contracts(distribution):
    dataset = _dataset(distribution, memories=6, seed=42)
    user = dataset.users(1, seed=19)[0]
    corpus = list(dataset.memory_stream(user))
    for scenario in (AddLoad(), SearchLoad(query_limit=3), Mixed(think=0)):
        ops = list(islice(scenario.plan(user, dataset, {"seed": 19}), 30))
        for op in ops:
            if op.query is not None:
                pool = corpus[:3] if scenario.name == "search-load" else corpus
                assert op.query.query in {item.content for item in pool}
            else:
                assert op.items[0] in corpus
        if scenario.name == "add-load":
            assert [op.items[0] for op in ops] == corpus * 5
    for namespace in ("first_", "second_"):
        wrapped = NamespacedDataset(_dataset(distribution, seed=42), namespace)
        mapped = wrapped.users(1, seed=7)[0]
        assert [item.content for item in islice(wrapped.memory_stream(mapped), 6)] == [
            item.content for item in corpus
        ]


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_hash_randomization_does_not_change_distributed_corpus(distribution):
    script = """
import json
from dataclasses import asdict
from ltm100.adapters.datasets.synthetic import SyntheticAdapter
ds = SyntheticAdapter(memories_per_user=12, categories=3, corpus_seed=42,
                      content_chars_distribution=json.loads(DISTRIBUTION))
user = ds.users(1, seed=19)[0]
print(json.dumps([asdict(item) for item in ds.memory_stream(user)], sort_keys=True))
""".replace("DISTRIBUTION", repr(json.dumps(distribution)))
    expected = json.dumps(
        [
            asdict(item)
            for item in _dataset(distribution, seed=42).memory_stream("syn_user_00000")
        ],
        sort_keys=True,
    )
    for hash_seed in ("0", "123"):
        output = subprocess.check_output(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            text=True,
        )
        assert output.strip() == expected


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_invalid_fixed_size_combination_including_explicit_default(distribution):
    for size in (200, 80, None):
        with pytest.raises(ValueError, match="mutually exclusive"):
            SyntheticAdapter(
                content_chars=size, content_chars_distribution=distribution
            )


def test_null_distribution_keeps_legacy_corpus_and_metadata():
    old = SyntheticAdapter(memories_per_user=5, content_chars=8)
    new = SyntheticAdapter(
        memories_per_user=5, content_chars=8, content_chars_distribution=None
    )
    assert new.generation_settings(seed=42) == old.generation_settings(seed=42)
    assert list(new.memory_stream("u")) == list(old.memory_stream("u"))


INVALID = [
    [],
    "uniform",
    0,
    False,
    {},
    {"type": "normal"},
    {"type": "uniform", "min_chars": 1},
    {"type": "uniform", "min_chars": 1, "max_chars": 8, "extra": 1},
    {"type": "uniform", "min_chars": 8, "max_chars": 1},
    {"type": "weighted"},
    {"type": "weighted", "choices": []},
    {"type": "weighted", "choices": "bad"},
    {"type": "weighted", "choices": [{"chars": 8}]},
    {"type": "weighted", "choices": [{"chars": 8, "weight": 1, "extra": 1}]},
    {
        "type": "weighted",
        "choices": [{"chars": 8, "weight": 1}, {"chars": 8, "weight": 2}],
    },
    {
        "type": "weighted",
        "choices": [{"chars": 8, "weight": 1e308}, {"chars": 9, "weight": 1e308}],
    },
]


@pytest.mark.parametrize("value", INVALID)
def test_malformed_distributions_fail_at_construction(value):
    with pytest.raises(ValueError, match="synthetic.content_chars_distribution"):
        _dataset(value)


@pytest.mark.parametrize("value", [0, -1, 1.0, 1.5, True, False, "8", None])
@pytest.mark.parametrize("field", ["min_chars", "max_chars", "chars"])
def test_distribution_requires_positive_integer_lengths(value, field):
    options = {"type": "uniform", "min_chars": 1, "max_chars": 8}
    if field == "chars":
        options = {"type": "weighted", "choices": [{"chars": value, "weight": 1}]}
    else:
        options[field] = value
    with pytest.raises(ValueError, match="integer >= 1"):
        _dataset(options)


@pytest.mark.parametrize(
    "weight", [0, -1, float("inf"), float("nan"), True, False, "1", None, 2**2048]
)
def test_distribution_rejects_invalid_weights(weight):
    with pytest.raises(ValueError, match="weight"):
        _dataset({"type": "weighted", "choices": [{"chars": 8, "weight": weight}]})


@pytest.mark.parametrize("procs", [1, 2])
def test_invalid_distribution_fails_before_backend_or_worker_work(
    tmp_path, monkeypatch, procs
):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid configuration reached the backend or workers")

    for name in (
        "_backend_build",
        "_probe_server_metrics",
        "run_shards",
        "build_backend",
    ):
        monkeypatch.setattr(cli, name, unexpected)
    options = {
        "content_chars_distribution": {
            "type": "uniform",
            "min_chars": 10,
            "max_chars": 1,
        }
    }
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
    with pytest.raises(ValueError, match="min_chars must be <= max_chars"):
        cli._run(args)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_cli_outputs_resolved_distribution_settings(
    tmp_path, monkeypatch, capsys, distribution
):
    options = {"content_chars_distribution": distribution, "corpus_seed": 42}
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
    assert (
        json.loads(capsys.readouterr().out.split("\nreports written to", 1)[0])
        == payload
    )
    assert payload["meta"]["seed"] == 19
    assert payload["meta"]["dataset_config"] == dataset_settings(
        AdapterConfig("synthetic", options), seed=19
    )


def test_invalid_distribution_sweep_fails_before_run_or_output(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid sweep reached a run")

    monkeypatch.setattr(cli, "_run", unexpected)
    output = tmp_path / "sweep"
    options = {"content_chars_distribution": {"type": "weighted", "choices": []}}
    args = cli.build_parser().parse_args(
        [
            "sweep",
            "memory-growth",
            "--config",
            _config(tmp_path, options),
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
    with pytest.raises(ValueError, match="non-empty list"):
        cli._run_memory_growth(args)
    assert not output.exists()
