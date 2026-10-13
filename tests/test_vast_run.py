"""Cost-cap and ownership logic of scripts/vast_run.py (no network calls)."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "vast_run", Path(__file__).resolve().parent.parent / "scripts" / "vast_run.py")
vast_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vast_run)

OFFER = {"id": 1, "machine_id": 42, "dph_total": 2.04, "storage_cost": 0.27, "cpu_arch": "amd64",
         "cuda_max_good": 13.2, "cpu_cores_effective": 24, "cpu_ram": 92 * 1024,
         "disk_space": 849}


def test_hourly_cost_adds_requested_disk():
    assert vast_run.hourly_cost(OFFER, disk_gb=40) == pytest.approx(2.04 + 40 * 0.27 / 720)


def test_lifetime_cap_matches_handoff_examples():
    # floor((C - R) * 3600 / H): C=5, R=0.5 -> 7200 s at $2.25/h, 5625 s at $2.88/h
    assert vast_run.max_lifetime_s(5, 2.25, reserve=0.5) == 7200
    assert vast_run.max_lifetime_s(5, 2.88, reserve=0.5) == 5625


def test_lifetime_cap_rejects_budget_below_reserve():
    with pytest.raises(ValueError):
        vast_run.max_lifetime_s(0.4, 2.0, reserve=0.5)


@pytest.mark.parametrize("field, bad", [("cuda_max_good", 12.2), ("cpu_arch", "arm64"),
                                        ("cpu_cores_effective", 2), ("cpu_ram", 8 * 1024),
                                        ("disk_space", 10)])
def test_ineligible_offers_are_filtered(field, bad):
    assert vast_run.eligible(OFFER)
    assert not vast_run.eligible(OFFER | {field: bad})


def test_ownership_requires_label_and_id():
    instances = [{"id": 7, "label": "digital-brain-a"}, {"id": 8, "label": "spectrans-n640"},
                 {"id": 9, "label": "digital-brain-a"}]
    assert [i["id"] for i in vast_run.owned(instances, "digital-brain-a", 7)] == [7]
    assert [i["id"] for i in vast_run.owned(instances, "digital-brain-a")] == [7, 9]
    assert vast_run.owned(instances, "digital-brain-b") == []


def test_pick_offer_matches_machine_within_rate():
    reissued = OFFER | {"id": 2}  # vast.ai re-issues offer ids for the same machine
    other = OFFER | {"id": 3, "machine_id": 43, "dph_total": 1.0}
    assert vast_run.pick_offer([other, reissued], 42, 2.10)["id"] == 2
    assert vast_run.pick_offer([other, reissued], 42, 2.00) is None  # over approved rate
    assert vast_run.pick_offer([other], 42, 9.0) is None
    cheap = OFFER | {"id": 4, "machine_id": 44, "dph_total": 1.5}
    assert vast_run.pick_offer([cheap, reissued], None, 2.10)["id"] == 4  # any machine
    assert vast_run.pick_offer([cheap, reissued], None, 1.0) is None


@pytest.mark.parametrize("geo, ok", [("Washington, US", True), (", US", True), ("US", True),
                                     ("Germany, DE", False), ("", False), (None, False)])
def test_country_filter(geo, ok):
    assert vast_run.in_country(OFFER | {"geolocation": geo}, "US") is ok
    assert vast_run.in_country(OFFER | {"geolocation": geo}, None)
