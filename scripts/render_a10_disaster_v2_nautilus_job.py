#!/usr/bin/env python3
"""Render and offline-verify Disaster V2 four-A10 Jobs."""

from __future__ import annotations

from typing import Sequence

import render_a10_nonvision_nautilus_job as renderer


renderer.WORKSPACE = "/workspace/perfseer-v3-native-a10-nonvision-11200-disaster-v2"
renderer.PROFILE = "native_a10_nonvision_disaster_v2"
renderer.JOB_PREFIX = "perfseer-v3-a10-nonvision-disaster-v2"
renderer.CAMPAIGN_LABEL = "native-a10-nonvision-disaster-11200-v2"


def main(argv: Sequence[str] | None = None) -> int:
    return renderer.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
