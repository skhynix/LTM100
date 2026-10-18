"""Opt-in disabled-server rejection preserves pre-existing tenant state."""

from __future__ import annotations

import os
import uuid

import pytest
import yaml

from ltm100.adapters.backends.memmachine import MemMachineClient
from ltm100.adapters.datasets.synthetic import SyntheticAdapter
from ltm100.cli import _run_shard_async, build_parser
from ltm100.common import MemoryItem, QueryItem

pytestmark = pytest.mark.skipif(
    not os.environ.get("LTM100_SEMANTIC_DISABLED_E2E_URL"),
    reason="requires a disposable MemMachine with semantic memory disabled",
)


async def test_live_disabled_semantic_preflight_preserves_existing_project(tmp_path):
    options = {
        "base_url": os.environ["LTM100_SEMANTIC_DISABLED_E2E_URL"],
        "org_prefix": f"types_disabled_{uuid.uuid4().hex}",
    }
    users = SyntheticAdapter().users(1, seed=0)
    async with MemMachineClient(**options) as baseline:
        await baseline.setup(users)
        try:
            await baseline.add(users[0], [MemoryItem("original corpus", role="user")])
            config = tmp_path / "config.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {"name": "synthetic"},
                        "backend": {
                            "name": "memmachine",
                            **options,
                            "search_memory_types": ["semantic"],
                        },
                    }
                )
            )
            args = build_parser().parse_args(
                [
                    "run",
                    "--config",
                    str(config),
                    "--scenario",
                    "search-load",
                    "--users",
                    "1",
                    "--ops",
                    "1",
                ]
            )
            with pytest.raises(ValueError, match="does not report enabled=true"):
                await _run_shard_async(args)
            rows = await baseline.search(users[0], QueryItem("original corpus"))
            assert [row.content for row in rows] == ["original corpus"]
        finally:
            await baseline.teardown(users, delete=True)
