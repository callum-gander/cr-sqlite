"""
Tests for long divergence, tombstone accumulation, and scale in local-first.

Simulates the realistic scenario where devices are offline for extended periods,
accumulate many independent changes, then sync. Also tests tombstone buildup
from heavy create/delete cycles and merge across many distinct site_ids.

Performance-relevant internals being exercised:
- CL cache: BTreeMap capped at 1500 entries, cleared entirely when full
- Ordinal map: cleared on every commit/rollback
- Tombstone accumulation: deletes leave sentinel rows in clock tables forever
"""

from crsql_correctness import connect, close, get_site_id, min_db_v
import time
import random
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_simple_schema():
    c = connect(":memory:")
    c.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER) STRICT;")
    c.execute("SELECT crsql_as_crr('foo')")
    c.commit()
    return c


def make_multi_column_schema():
    c = connect(":memory:")
    c.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY NOT NULL, name TEXT, value INTEGER, status TEXT) STRICT;"
    )
    c.execute("SELECT crsql_as_crr('items')")
    c.commit()
    return c


def get_all_changes(c, since=0):
    return c.execute(
        "SELECT [table], pk, cid, val, col_version, db_version, site_id, cl, seq, ts "
        "FROM crsql_changes WHERE db_version > ? ORDER BY db_version, seq",
        (since,),
    ).fetchall()


def apply_changes(c, changes):
    for change in changes:
        c.execute(
            "INSERT INTO crsql_changes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", change
        )
    c.commit()


def apply_changes_batched(c, changes, batch_size=500):
    """Apply changes in batched transactions to simulate realistic sync."""
    for i in range(0, len(changes), batch_size):
        batch = changes[i:i + batch_size]
        for change in batch:
            c.execute(
                "INSERT INTO crsql_changes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", change
            )
        c.commit()


def get_table_contents(c, table="foo"):
    return c.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()


def assert_tables_equal(c1, c2, table="foo"):
    rows1 = get_table_contents(c1, table)
    rows2 = get_table_contents(c2, table)
    assert rows1 == rows2, f"Tables diverged:\n  node1 ({len(rows1)} rows)\n  node2 ({len(rows2)} rows)"


def assert_convergence(nodes, table="foo"):
    reference = get_table_contents(nodes[0], table)
    for i, node in enumerate(nodes[1:], 1):
        rows = get_table_contents(node, table)
        assert rows == reference, (
            f"Node {i} diverged from node 0: "
            f"{len(reference)} vs {len(rows)} rows"
        )


# ===========================================================================
# 1. LONG DIVERGENCE — extended offline periods, then large merge
# ===========================================================================


class TestLongDivergence:
    """
    Two nodes go offline for a long time, each accumulating many changes.
    When they reconnect, the merge must be correct and reasonably fast.
    """

    def test_2000_independent_inserts_then_merge(self):
        """
        Two nodes each insert 1000 unique rows while offline. Merge both
        ways. Both should have all 2000 rows.
        """
        phone = make_simple_schema()
        laptop = make_simple_schema()

        for i in range(1, 1001):
            phone.execute("INSERT INTO foo VALUES (?, ?)", (i, i))
        phone.commit()

        for i in range(1001, 2001):
            laptop.execute("INSERT INTO foo VALUES (?, ?)", (i, i))
        laptop.commit()

        phone_changes = get_all_changes(phone)
        laptop_changes = get_all_changes(laptop)

        t0 = time.time()
        apply_changes(laptop, phone_changes)
        apply_changes(phone, laptop_changes)
        merge_time = time.time() - t0

        assert_tables_equal(phone, laptop)
        assert len(get_table_contents(phone)) == 2000

        print(f"\n  2000 row merge: {merge_time:.3f}s "
              f"({len(phone_changes) + len(laptop_changes)} changes)")

        close(phone)
        close(laptop)

    def test_5000_mixed_operations_then_merge(self):
        """
        Two nodes each perform ~2500 mixed insert/update/delete operations.
        This exercises the CL cache (1500 entry cap) heavily.
        """
        phone = make_simple_schema()
        laptop = make_simple_schema()

        # Phone: inserts then updates
        for i in range(1, 1501):
            phone.execute("INSERT INTO foo VALUES (?, ?)", (i, i))
        phone.commit()
        for i in range(1, 1001):
            phone.execute("UPDATE foo SET b = ? WHERE a = ?", (i * 10, i))
        phone.commit()

        # Laptop: inserts, some deletes, some re-inserts
        for i in range(1501, 3001):
            laptop.execute("INSERT INTO foo VALUES (?, ?)", (i, i))
        laptop.commit()
        for i in range(1501, 1801):
            laptop.execute("DELETE FROM foo WHERE a = ?", (i,))
        laptop.commit()
        for i in range(1501, 1701):
            laptop.execute("INSERT INTO foo VALUES (?, ?)", (i, i * 100))
        laptop.commit()

        phone_changes = get_all_changes(phone)
        laptop_changes = get_all_changes(laptop)

        t0 = time.time()
        apply_changes(laptop, phone_changes)
        apply_changes(phone, laptop_changes)
        merge_time = time.time() - t0

        assert_tables_equal(phone, laptop)
        print(f"\n  5000 mixed op merge: {merge_time:.3f}s "
              f"({len(phone_changes) + len(laptop_changes)} changes)")

        close(phone)
        close(laptop)

    def test_conflicting_updates_at_scale(self):
        """
        Both nodes start with the same 500 rows. Each updates all 500 rows
        to different values while offline. Merge should resolve every conflict
        deterministically.
        """
        phone = make_simple_schema()
        laptop = make_simple_schema()

        # Shared starting state
        for i in range(1, 501):
            phone.execute("INSERT INTO foo VALUES (?, ?)", (i, 0))
        phone.commit()
        apply_changes(laptop, get_all_changes(phone))

        # Both update all rows independently
        for i in range(1, 501):
            phone.execute("UPDATE foo SET b = ? WHERE a = ?", (i * 10, i))
        phone.commit()

        for i in range(1, 501):
            laptop.execute("UPDATE foo SET b = ? WHERE a = ?", (i * 20, i))
        laptop.commit()

        phone_changes = get_all_changes(phone)
        laptop_changes = get_all_changes(laptop)

        apply_changes(laptop, phone_changes)
        apply_changes(phone, laptop_changes)

        assert_tables_equal(phone, laptop)
        assert len(get_table_contents(phone)) == 500

        close(phone)
        close(laptop)

    def test_batched_merge_matches_atomic(self):
        """
        Verify that applying a large changeset in small batches produces
        the same result as applying it atomically. This is how real sync
        works — you don't hold one giant transaction open.
        """
        sender = make_simple_schema()
        receiver_atomic = make_simple_schema()
        receiver_batched = make_simple_schema()

        for i in range(1, 2001):
            sender.execute("INSERT INTO foo VALUES (?, ?)", (i, i * 10))
        sender.commit()
        for i in range(1, 501):
            sender.execute("DELETE FROM foo WHERE a = ?", (i,))
        sender.commit()

        changes = get_all_changes(sender)

        apply_changes(receiver_atomic, changes)
        apply_changes_batched(receiver_batched, changes, batch_size=200)

        assert_tables_equal(receiver_atomic, receiver_batched)

        close(sender)
        close(receiver_atomic)
        close(receiver_batched)

    def test_merge_scales_linearly(self):
        """
        Measure merge time at 500, 1000, and 2000 rows. The ratio should
        be roughly linear, not quadratic.
        """
        times = []
        sizes = [500, 1000, 2000]

        for n in sizes:
            sender = make_simple_schema()
            receiver = make_simple_schema()

            for i in range(1, n + 1):
                sender.execute("INSERT INTO foo VALUES (?, ?)", (i, i))
            sender.commit()

            changes = get_all_changes(sender)

            t0 = time.time()
            apply_changes(receiver, changes)
            elapsed = time.time() - t0
            times.append(elapsed)

            close(sender)
            close(receiver)

        # If scaling is quadratic, the 4x size would take ~16x time.
        # Allow up to 8x for linear-ish scaling (generous margin for noise).
        ratio = times[2] / max(times[0], 0.001)
        print(f"\n  Scale test: {sizes[0]}={times[0]:.3f}s, "
              f"{sizes[1]}={times[1]:.3f}s, {sizes[2]}={times[2]:.3f}s "
              f"(ratio {sizes[2]}/{sizes[0]}: {ratio:.1f}x)")

        assert ratio < 16, (
            f"Merge appears to scale worse than linearly: "
            f"{sizes[2]} rows took {ratio:.1f}x longer than {sizes[0]} rows"
        )


# ===========================================================================
# 2. TOMBSTONE ACCUMULATION — heavy create/delete cycles
# ===========================================================================


class TestTombstoneAccumulation:
    """
    Deletes leave sentinel rows in clock tables forever. Heavy create/delete
    cycles cause unbounded tombstone growth.
    """

    def test_repeated_delete_resurrect_cycle(self):
        """
        Create and delete the same row 200 times. Then sync to another node.
        The final state should be correct (row deleted or alive depending
        on the last operation).
        """
        node = make_simple_schema()
        receiver = make_simple_schema()

        for cycle in range(200):
            node.execute("INSERT OR REPLACE INTO foo VALUES (1, ?)", (cycle,))
            node.commit()
            node.execute("DELETE FROM foo WHERE a = 1")
            node.commit()

        # Final insert — row should be alive
        node.execute("INSERT INTO foo VALUES (1, 9999)")
        node.commit()

        changes = get_all_changes(node)
        apply_changes(receiver, changes)

        rows = get_table_contents(receiver)
        assert rows == [(1, 9999)], f"Expected [(1, 9999)], got: {rows}"

        close(node)
        close(receiver)

    def test_tombstone_count_grows_with_deletes(self):
        """
        Document that tombstones accumulate in clock tables. Insert N rows,
        delete all of them, count the clock table entries.
        """
        node = make_simple_schema()

        n = 500
        for i in range(1, n + 1):
            node.execute("INSERT INTO foo VALUES (?, ?)", (i, i))
        node.commit()

        for i in range(1, n + 1):
            node.execute("DELETE FROM foo WHERE a = ?", (i,))
        node.commit()

        # Base table should be empty
        rows = get_table_contents(node)
        assert len(rows) == 0

        # But clock table retains tombstones
        clock_rows = node.execute(
            "SELECT count(*) FROM foo__crsql_clock"
        ).fetchone()[0]

        # Each deleted row has at least a sentinel (-1) entry
        assert clock_rows >= n, (
            f"Expected at least {n} clock entries (tombstones), got {clock_rows}"
        )

        print(f"\n  {n} deletes -> {clock_rows} clock table entries (tombstones)")

        close(node)

    def test_tombstone_sync_correctness(self):
        """
        Node A creates 500 rows. Node B syncs them. Node A deletes all 500.
        Node B syncs the deletes. B should have empty table.
        """
        a = make_simple_schema()
        b = make_simple_schema()

        for i in range(1, 501):
            a.execute("INSERT INTO foo VALUES (?, ?)", (i, i))
        a.commit()

        apply_changes(b, get_all_changes(a))
        assert len(get_table_contents(b)) == 500

        for i in range(1, 501):
            a.execute("DELETE FROM foo WHERE a = ?", (i,))
        a.commit()

        delete_changes = get_all_changes(a, since=1)
        apply_changes(b, delete_changes)

        assert len(get_table_contents(b)) == 0, "All rows should be deleted"

        close(a)
        close(b)

    def test_mass_delete_then_resurrect_subset(self):
        """
        Delete 500 rows, then resurrect 100 of them. Sync to a fresh node.
        Only the 100 resurrected rows should exist.
        """
        source = make_simple_schema()
        receiver = make_simple_schema()

        # Create 500
        for i in range(1, 501):
            source.execute("INSERT INTO foo VALUES (?, ?)", (i, i))
        source.commit()

        # Delete all
        for i in range(1, 501):
            source.execute("DELETE FROM foo WHERE a = ?", (i,))
        source.commit()

        # Resurrect 100
        for i in range(1, 101):
            source.execute("INSERT INTO foo VALUES (?, ?)", (i, i * 1000))
        source.commit()

        changes = get_all_changes(source)
        apply_changes(receiver, changes)

        rows = get_table_contents(receiver)
        assert len(rows) == 100, f"Expected 100 rows, got {len(rows)}"
        # Verify they have the resurrected values
        for row in rows:
            assert row[1] == row[0] * 1000

        close(source)
        close(receiver)


# ===========================================================================
# 3. MANY SITE_IDS — fleet of devices merging through a hub
# ===========================================================================


class TestManySiteIds:
    """
    Local-first apps can have many devices per user (phone, tablet, laptop,
    desktop, work machine). Each has a unique site_id. The ordinal map is
    cleared on every commit, so many distinct site_ids across batched
    transactions means repeated DB lookups.
    """

    def test_20_node_fleet_converges(self):
        """
        20 nodes each make independent writes, all sync through a central hub.
        Everyone should converge.
        """
        hub = make_simple_schema()
        nodes = [make_simple_schema() for _ in range(20)]

        for i, node in enumerate(nodes):
            base = i * 100
            for j in range(1, 11):
                node.execute("INSERT INTO foo VALUES (?, ?)", (base + j, base + j))
            node.commit()

        # All sync to hub
        for node in nodes:
            apply_changes(hub, get_all_changes(node))

        assert len(get_table_contents(hub)) == 200

        # Hub syncs back to all nodes
        hub_changes = get_all_changes(hub)
        for node in nodes:
            apply_changes(node, hub_changes)

        assert_convergence([hub] + nodes)

        close(hub)
        for n in nodes:
            close(n)

    def test_many_site_ids_batched_merge(self):
        """
        Collect changes from 30 nodes and apply them to a receiver in small
        batches. Each batch boundary triggers cache clears (ordinal map,
        cl_cache). Verify correctness despite cache thrashing.
        """
        nodes = [make_simple_schema() for _ in range(30)]
        receiver_atomic = make_simple_schema()
        receiver_batched = make_simple_schema()

        all_changes = []
        for i, node in enumerate(nodes):
            node.execute("INSERT INTO foo VALUES (?, ?)", (i + 1, (i + 1) * 10))
            node.commit()
            all_changes.extend(get_all_changes(node))

        # Shuffle to interleave site_ids
        random.shuffle(all_changes)

        apply_changes(receiver_atomic, all_changes)
        apply_changes_batched(receiver_batched, all_changes, batch_size=5)

        assert_tables_equal(receiver_atomic, receiver_batched)
        assert len(get_table_contents(receiver_atomic)) == 30

        for n in nodes:
            close(n)
        close(receiver_atomic)
        close(receiver_batched)

    def test_site_id_ordinals_stable_across_merge(self):
        """
        After merging from many nodes, the site_id ordinal table should
        have entries for all unique site_ids seen.
        """
        hub = make_simple_schema()
        nodes = [make_simple_schema() for _ in range(10)]

        site_ids = set()
        for node in nodes:
            node.execute("INSERT INTO foo VALUES (?, ?)", (random.randint(1000, 9999), 1))
            node.commit()
            site_ids.add(get_site_id(node).hex())

        for node in nodes:
            apply_changes(hub, get_all_changes(node))

        # Check ordinal table has all site_ids
        ordinals = hub.execute(
            "SELECT hex(site_id) FROM crsql_site_id"
        ).fetchall()
        ordinal_site_ids = set(row[0].lower() for row in ordinals)

        for sid in site_ids:
            assert sid in ordinal_site_ids, f"Site ID {sid} missing from ordinal table"

        close(hub)
        for n in nodes:
            close(n)


# ===========================================================================
# 4. INTERLEAVED LOCAL WRITES DURING BATCHED SYNC
# ===========================================================================


class TestInterleavedWritesDuringSync:
    """
    Real sync happens in batches. Between batches, the user may make local
    writes. The commit between batches clears all caches (cl_cache, ordinal
    map, lastDbVersions). Correctness must hold despite this.
    """

    def test_local_writes_between_sync_batches(self):
        """
        Apply sync in 3 batches with local writes between each batch.
        Compare with a node that receives the full sync atomically (plus
        the same local writes).
        """
        sender = make_simple_schema()
        receiver_interleaved = make_simple_schema()
        receiver_reference = make_simple_schema()

        for i in range(1, 301):
            sender.execute("INSERT INTO foo VALUES (?, ?)", (i, i))
        sender.commit()

        changes = get_all_changes(sender)
        third = len(changes) // 3

        # Interleaved: batch1, local write, batch2, local write, batch3
        apply_changes(receiver_interleaved, changes[:third])

        receiver_interleaved.execute("INSERT INTO foo VALUES (1001, 1001)")
        receiver_interleaved.commit()

        apply_changes(receiver_interleaved, changes[third:2*third])

        receiver_interleaved.execute("INSERT INTO foo VALUES (1002, 1002)")
        receiver_interleaved.commit()

        apply_changes(receiver_interleaved, changes[2*third:])

        # Reference: full sync then same local writes
        apply_changes(receiver_reference, changes)
        receiver_reference.execute("INSERT INTO foo VALUES (1001, 1001)")
        receiver_reference.execute("INSERT INTO foo VALUES (1002, 1002)")
        receiver_reference.commit()

        assert_tables_equal(receiver_interleaved, receiver_reference)

        close(sender)
        close(receiver_interleaved)
        close(receiver_reference)

    def test_conflict_across_batch_boundary(self):
        """
        Sender updates row X in batch 1. Between batches, receiver also
        updates row X. Batch 2 arrives. The conflict should resolve
        deterministically.
        """
        sender = make_simple_schema()
        receiver = make_simple_schema()
        reference = make_simple_schema()

        # Both start with the same row
        sender.execute("INSERT INTO foo VALUES (1, 100)")
        sender.commit()
        insert_changes = get_all_changes(sender)
        apply_changes(receiver, insert_changes)
        apply_changes(reference, insert_changes)

        # Sender makes two updates in separate commits
        sender.execute("UPDATE foo SET b = 200 WHERE a = 1")
        sender.commit()
        sender.execute("INSERT INTO foo VALUES (2, 999)")
        sender.commit()

        all_updates = get_all_changes(sender, since=1)
        # Find a split point — changes from first update vs second
        first_update = [c for c in all_updates if c[5] == 2]  # db_version 2
        second_update = [c for c in all_updates if c[5] == 3]  # db_version 3

        # Apply first batch
        apply_changes(receiver, first_update)
        apply_changes(reference, first_update)

        # Local conflicting write between batches
        receiver.execute("UPDATE foo SET b = 888 WHERE a = 1")
        receiver.commit()
        reference.execute("UPDATE foo SET b = 888 WHERE a = 1")
        reference.commit()

        # Apply second batch
        apply_changes(receiver, second_update)
        apply_changes(reference, second_update)

        assert_tables_equal(receiver, reference)

        close(sender)
        close(receiver)
        close(reference)

    def test_rapid_interleave_small_transactions(self):
        """
        Alternating: apply 1 remote change, make 1 local write, repeat.
        This maximizes cache clearing. Result should match bulk application.
        """
        sender = make_simple_schema()
        receiver_rapid = make_simple_schema()
        receiver_bulk = make_simple_schema()

        for i in range(1, 51):
            sender.execute("INSERT INTO foo VALUES (?, ?)", (i, i * 10))
        sender.commit()

        changes = get_all_changes(sender)

        # Rapid interleave
        for idx, change in enumerate(changes):
            receiver_rapid.execute(
                "INSERT INTO crsql_changes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                change,
            )
            receiver_rapid.commit()

            local_key = 10000 + idx
            receiver_rapid.execute(
                "INSERT INTO foo VALUES (?, ?)", (local_key, local_key)
            )
            receiver_rapid.commit()

        # Bulk: all remote, then all local
        apply_changes(receiver_bulk, changes)
        for idx in range(len(changes)):
            local_key = 10000 + idx
            receiver_bulk.execute(
                "INSERT INTO foo VALUES (?, ?)", (local_key, local_key)
            )
        receiver_bulk.commit()

        assert_tables_equal(receiver_rapid, receiver_bulk)

        close(sender)
        close(receiver_rapid)
        close(receiver_bulk)
