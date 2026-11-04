"""
Lab 6 - Branch coverage tests for cloud_storage_tier.py

Goal: make every decision (if / elif / else / for / break / continue) evaluate to BOTH
True and False at least once (>= 90% branch coverage), including boundary values.

Run:
    coverage run --branch -m pytest BranchCoverageTest_cloud_storage_tier.py
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


def bucket(size_gb=100, corrupted=False, egress_gb=0):
    return {"size_gb": size_gb, "corrupted": corrupted, "egress_gb": egress_gb}


# =====================================================================
# process_storage_tenants
# =====================================================================

# ---- Branch: input validation (isinstance(...) False/True, fee < 0 True/False) ----

@pytest.mark.parametrize("tenants, fee", [
    ("not-a-list", 10.0),          # first condition True
    ({"a": 1}, 10.0),              # dict is not a list
    (None, 10.0),                  # None is not a list
    ([], -1.0),                    # second condition True (valid list, negative fee)
    ("bad", -5.0),                 # both conditions True
])
def test_invalid_input_raises(tenants, fee):
    with pytest.raises(ValueError, match="Invalid tenants list or support fee"):
        process_storage_tenants(tenants, fee)


def test_empty_list_and_zero_fee_is_valid():
    # fee == 0 -> (fee < 0) False, list valid -> no raise; empty loop body never entered
    assert process_storage_tenants([], 0) == {"total_billed": 0.0, "status": "ZERO_USAGE"}


# ---- Branch: account_active True / False / missing ----

def test_inactive_tenant_is_skipped():
    result = process_storage_tenants([make_tenant(active=False, buckets=[bucket(100)])], 5.0)
    assert result == {"total_billed": 0.0, "status": "ZERO_USAGE"}


def test_missing_account_active_defaults_to_inactive():
    result = process_storage_tenants([{"tier": "hot", "buckets": [bucket(100)]}], 5.0)
    assert result == {"total_billed": 0.0, "status": "ZERO_USAGE"}


def test_mixed_active_and_inactive_tenants():
    tenants = [make_tenant(active=False, buckets=[bucket(1000)]),
               make_tenant("hot", [bucket(100)])]
    result = process_storage_tenants(tenants, 0.0)
    assert result["total_billed"] == pytest.approx(2.5)
    assert result["status"] == "BILLED"


# ---- Branch: tier selection (hot / cool / archive / unknown / default) ----

def test_hot_tier_rate():
    r = process_storage_tenants([make_tenant("hot", [bucket(100)])], 0.0)
    assert r["total_billed"] == pytest.approx(2.5)


def test_cool_tier_rate():
    r = process_storage_tenants([make_tenant("cool", [bucket(200)])], 0.0)
    assert r["total_billed"] == pytest.approx(3.0)


def test_archive_tier_rate():
    r = process_storage_tenants([make_tenant("archive", [bucket(200)])], 0.0)
    assert r["total_billed"] == pytest.approx(1.0)


def test_unknown_tier_is_flagged_and_skipped():
    r = process_storage_tenants([make_tenant("platinum", [bucket(100)], vault=True)], 50.0)
    # skipped via `continue` -> neither storage nor vault fee is billed
    assert r == {"total_billed": 0.0, "status": "ZERO_USAGE"}


def test_missing_tier_defaults_to_hot_and_missing_buckets_defaults_to_empty():
    # no "tier" key -> default "hot"; no "buckets" key -> default [] (inner loop skipped)
    r = process_storage_tenants([{"account_active": True, "has_backup_vault": True}], 7.0)
    assert r == {"total_billed": 7.0, "status": "BILLED"}


def test_missing_tier_with_bucket_uses_hot_rate():
    r = process_storage_tenants([{"account_active": True, "buckets": [bucket(100)]}], 0.0)
    assert r["total_billed"] == pytest.approx(2.5)


# ---- Branch: bucket size (<= 0, normal, > 500) with boundaries ----

@pytest.mark.parametrize("size", [0, -10])
def test_non_positive_bucket_size_is_flagged(size):
    r = process_storage_tenants([make_tenant("hot", [bucket(size)])], 0.0)
    assert r == {"total_billed": 0.0, "status": "ZERO_USAGE"}


def test_missing_size_defaults_to_zero_and_is_flagged():
    r = process_storage_tenants([make_tenant("hot", [{"corrupted": False}])], 0.0)
    assert r["status"] == "ZERO_USAGE"


def test_size_exactly_500_gets_no_discount():
    r = process_storage_tenants([make_tenant("hot", [bucket(500)])], 0.0)
    assert r["total_billed"] == pytest.approx(12.5)


def test_size_501_gets_ten_percent_discount():
    r = process_storage_tenants([make_tenant("hot", [bucket(501)])], 0.0)
    assert r["total_billed"] == pytest.approx(11.27)  # 501 * 0.025 * 0.90 = 11.2725, rounded to 2 dp


def test_size_smallest_positive_is_billed():
    r = process_storage_tenants([make_tenant("hot", [bucket(2)])], 0.0)
    assert r["total_billed"] == pytest.approx(0.05)


# ---- Branch: corrupted bucket (non-archive vs archive) ----

def test_corrupted_bucket_on_hot_tier_not_charged_for_storage():
    r = process_storage_tenants([make_tenant("hot", [bucket(100, corrupted=True)])], 0.0)
    assert r == {"total_billed": 0.0, "status": "ZERO_USAGE"}


def test_corrupted_bucket_on_cool_tier_does_not_stop_processing():
    # corrupted bucket is skipped for storage, next healthy bucket is still billed
    r = process_storage_tenants(
        [make_tenant("cool", [bucket(100, corrupted=True), bucket(200)])], 0.0)
    assert r["total_billed"] == pytest.approx(3.0)


def test_corrupted_archive_bucket_resets_charge_and_breaks():
    # first bucket billed (1.0), second corrupted -> charge reset to 0 and loop breaks,
    # the third bucket must never be processed.
    tenants = [make_tenant("archive", [bucket(200), bucket(100, corrupted=True), bucket(1000)])]
    r = process_storage_tenants(tenants, 0.0)
    assert r == {"total_billed": 0.0, "status": "ZERO_USAGE"}


def test_corrupted_archive_still_pays_backup_vault_fee():
    # break only exits the bucket loop; vault fee is added afterwards
    tenants = [make_tenant("archive", [bucket(100, corrupted=True)], vault=True)]
    r = process_storage_tenants(tenants, 7.0)
    assert r == {"total_billed": 7.0, "status": "BILLED"}


# ---- Branch: egress (> 100 vs <= 100) with boundaries ----

def test_egress_exactly_100_is_free():
    r = process_storage_tenants([make_tenant("hot", [bucket(100, egress_gb=100)])], 0.0)
    assert r["total_billed"] == pytest.approx(2.5)


def test_egress_101_is_charged_for_one_gb():
    r = process_storage_tenants([make_tenant("hot", [bucket(100, egress_gb=101)])], 0.0)
    assert r["total_billed"] == pytest.approx(2.5 + 0.08)


def test_egress_over_100_on_large_bucket():
    r = process_storage_tenants([make_tenant("hot", [bucket(1000, egress_gb=200)])], 0.0)
    assert r["total_billed"] == pytest.approx(22.5 + 100 * 0.08)


def test_egress_is_charged_even_for_corrupted_non_archive_bucket():
    r = process_storage_tenants([make_tenant("hot", [bucket(100, corrupted=True, egress_gb=150)])], 0.0)
    assert r["total_billed"] == pytest.approx(50 * 0.08)
    assert r["status"] == "BILLED"


def test_egress_skipped_for_invalid_size_bucket():
    # `continue` on size <= 0 happens before egress is evaluated
    r = process_storage_tenants([make_tenant("hot", [bucket(0, egress_gb=500)])], 0.0)
    assert r["total_billed"] == 0.0


# ---- Branch: backup vault True / False ----

def test_backup_vault_fee_added():
    r = process_storage_tenants([make_tenant("hot", [bucket(100)], vault=True)], 10.0)
    assert r["total_billed"] == pytest.approx(12.5)


def test_no_backup_vault_no_fee():
    r = process_storage_tenants([make_tenant("hot", [bucket(100)], vault=False)], 10.0)
    assert r["total_billed"] == pytest.approx(2.5)


def test_vault_with_zero_fee():
    r = process_storage_tenants([make_tenant("hot", [bucket(100)], vault=True)], 0.0)
    assert r["total_billed"] == pytest.approx(2.5)


# ---- Branch: final status (AUDIT_FLAGGED / ZERO_USAGE / BILLED) ----

def test_exactly_four_flagged_buckets_is_not_audit():
    r = process_storage_tenants([make_tenant("hot", [bucket(0)] * 4 + [bucket(100)])], 0.0)
    assert r["status"] == "BILLED"
    assert r["total_billed"] == pytest.approx(2.5)


def test_five_flagged_buckets_triggers_audit():
    r = process_storage_tenants([make_tenant("hot", [bucket(0)] * 5 + [bucket(100)])], 0.0)
    assert r == {"total_billed": 2.5, "status": "AUDIT_FLAGGED"}


def test_flags_accumulate_across_tenants_and_types():
    tenants = [
        make_tenant("platinum", [bucket(10)]),            # 1 (unknown tier)
        make_tenant("hot", [bucket(100, corrupted=True)]),  # 2 (corrupted)
        make_tenant("cool", [bucket(0)]),                 # 3 (bad size)
        make_tenant("archive", [bucket(50, corrupted=True)]),  # 4 (corrupted archive)
        make_tenant("hot", [bucket(-1)]),                 # 5 (bad size)
    ]
    assert process_storage_tenants(tenants, 0.0)["status"] == "AUDIT_FLAGGED"


def test_audit_takes_priority_over_zero_usage():
    r = process_storage_tenants([make_tenant("hot", [bucket(0)] * 5)], 0.0)
    assert r == {"total_billed": 0.0, "status": "AUDIT_FLAGGED"}


def test_zero_usage_with_no_active_tenants():
    assert process_storage_tenants([make_tenant(active=False)], 10.0) == \
        {"total_billed": 0.0, "status": "ZERO_USAGE"}


def test_multiple_tenants_are_summed_and_rounded():
    tenants = [
        make_tenant("hot", [bucket(100, egress_gb=150)], vault=True),   # 2.5 + 4.0 + 3.333
        make_tenant("cool", [bucket(1000)]),                            # 15 * 0.9 = 13.5
        make_tenant("archive", [bucket(333)]),                          # 1.665
    ]
    r = process_storage_tenants(tenants, 3.333)
    assert r["status"] == "BILLED"
    assert r["total_billed"] == round(2.5 + 4.0 + 3.333 + 13.5 + 1.665, 2)


# =====================================================================
# monitor_replication_latency
# =====================================================================

@pytest.mark.parametrize("matrix", [[], None])
def test_monitor_no_data(matrix):
    assert monitor_replication_latency(matrix) == "NO_DATA"


def test_monitor_node_with_empty_logs_is_synced():
    # outer loop entered, inner loop never entered
    assert monitor_replication_latency([[]]) == "SYNCED"


def test_monitor_all_below_threshold():
    assert monitor_replication_latency([[10, 50, 100], [20, 30]]) == "SYNCED"


def test_monitor_exactly_200_is_not_a_spike():
    assert monitor_replication_latency([[200, 200, 200]]) == "SYNCED"


def test_monitor_single_spike_is_not_enough():
    assert monitor_replication_latency([[100, 500, 100]]) == "SYNCED"


def test_monitor_non_consecutive_spikes_reset_streak():
    assert monitor_replication_latency([[250, 100, 250, 100, 250]]) == "SYNCED"


def test_monitor_two_consecutive_spikes_alert():
    assert monitor_replication_latency([[201, 201]]) == "SYNC_LAG_ALERT"


def test_monitor_alert_after_streak_reset():
    assert monitor_replication_latency([[250, 100, 250, 250]]) == "SYNC_LAG_ALERT"


def test_monitor_alert_on_later_node():
    assert monitor_replication_latency([[10, 20], [300, 400]]) == "SYNC_LAG_ALERT"


def test_monitor_streak_does_not_carry_over_between_nodes():
    # one spike at the end of node 1 and one at the start of node 2 are NOT consecutive
    assert monitor_replication_latency([[100, 250], [250, 100]]) == "SYNCED"


def test_monitor_alert_returns_early():
    # the alert on the first node must stop processing (second node never needed)
    assert monitor_replication_latency([[300, 300], [10]]) == "SYNC_LAG_ALERT"
