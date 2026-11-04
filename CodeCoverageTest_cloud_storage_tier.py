"""
Lab 6 - Code (line) coverage tests for cloud_storage_tier.py

Goal: execute every line of the module at least once (>= 90% line coverage).

Run:
    coverage run -m pytest CodeCoverageTest_cloud_storage_tier.py
    coverage report -m
"""
import pytest
from cloud_storage_tier import process_storage_tenants, monitor_replication_latency


def make_tenant(tier="hot", buckets=None, active=True, vault=False):
    return {
        "account_active": active,
        "tier": tier,
        "has_backup_vault": vault,
        "buckets": buckets if buckets is not None else [],
    }


# ---------------------------------------------------------------- process_storage_tenants

def test_single_active_hot_tenant():
    # Starter example: 100 GB * 0.025 = 2.5
    tenants = [make_tenant("hot", [{"size_gb": 100, "corrupted": False, "egress_gb": 50}])]
    result = process_storage_tenants(tenants, 10.0)
    assert result["total_billed"] == 2.5
    assert result["status"] == "BILLED"


def test_invalid_input_raises():
    # line 3: raise ValueError
    with pytest.raises(ValueError):
        process_storage_tenants("not-a-list", 10.0)


def test_unknown_tier_flagged():
    # lines 22-23: unknown tier -> flagged and skipped
    result = process_storage_tenants([make_tenant("platinum", [{"size_gb": 10}])], 0.0)
    assert result["status"] == "ZERO_USAGE"


def test_invalid_bucket_size_flagged():
    # lines 33-34: size_gb <= 0 -> flagged, skipped
    tenants = [make_tenant("hot", [{"size_gb": 0}])]
    result = process_storage_tenants(tenants, 0.0)
    assert result["total_billed"] == 0.0


def test_corrupted_bucket_not_billed():
    # lines 37-38, 41: corrupted bucket on non-archive tier adds nothing
    tenants = [make_tenant("hot", [{"size_gb": 100, "corrupted": True}])]
    result = process_storage_tenants(tenants, 0.0)
    assert result["total_billed"] == 0.0


def test_corrupted_archive_bucket_zeroes_tenant():
    # lines 39-40: corrupted bucket on archive tier resets the charge and breaks
    tenants = [make_tenant("archive", [
        {"size_gb": 100, "corrupted": False},
        {"size_gb": 100, "corrupted": True},
    ])]
    result = process_storage_tenants(tenants, 0.0)
    assert result["total_billed"] == 0.0


def test_large_bucket_discount():
    # line 43: size_gb > 500 -> 10% discount
    tenants = [make_tenant("hot", [{"size_gb": 1000, "corrupted": False}])]
    result = process_storage_tenants(tenants, 0.0)
    assert result["total_billed"] == pytest.approx(22.5)


def test_egress_charge():
    # line 48: egress above 100 GB billed at 0.08
    tenants = [make_tenant("hot", [{"size_gb": 100, "egress_gb": 150}])]
    result = process_storage_tenants(tenants, 0.0)
    assert result["total_billed"] == pytest.approx(2.5 + 50 * 0.08)


def test_audit_flagged():
    # line 56: more than 4 flagged buckets
    tenants = [make_tenant("hot", [{"size_gb": 0}] * 5)]
    result = process_storage_tenants(tenants, 0.0)
    assert result["status"] == "AUDIT_FLAGGED"


# ---------------------------------------------------------------- monitor_replication_latency

def test_monitor_no_data():
    # line 63
    assert monitor_replication_latency([]) == "NO_DATA"


def test_monitor_sync_lag_alert():
    # lines 66-72: two consecutive latencies > 200 ms
    assert monitor_replication_latency([[100, 250, 300]]) == "SYNC_LAG_ALERT"
