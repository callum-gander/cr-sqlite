"""
Tests for local-first sync scenarios.

The existing test suite assumes perfect, atomic, complete sync transfers between
controlled cloud nodes. These tests exercise what happens when sync is unreliable —
partial delivery, duplicate delivery, out-of-order delivery — as it would be between
user-owned devices (phones, laptops, tablets) on flaky networks.
"""

from crsql_correctness import connect, close, get_site_id, min_db_v
import random
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis.strategies import integers, data, lists, permutations


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
    """Get all changes from a node as a list of tuples."""
    return c.execute(
        "SELECT [table], pk, cid, val, col_version, db_version, site_id, cl, seq, ts "
        "FROM crsql_changes WHERE db_version > ? ORDER BY db_version, seq",
        (since,),
    ).fetchall()


def get_all_changes_excluding_site(c, exclude_site_id, since=0):
    """Get changes excluding our own site_id (typical delta-state sync)."""
    return c.execute(
        "SELECT [table], pk, cid, val, col_version, db_version, site_id, cl, seq, ts "
        "FROM crsql_changes WHERE db_version > ? AND site_id IS NOT ? ORDER BY db_version, seq",
        (since, exclude_site_id),
    ).fetchall()


def apply_changes(c, changes):
    """Apply a list of changes to a node and commit."""
    for change in changes:
        c.execute(
            "INSERT INTO crsql_changes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", change
        )
    c.commit()


def apply_changes_no_commit(c, changes):
    """Apply changes without committing (for testing rollback scenarios)."""
    for change in changes:
        c.execute(
            "INSERT INTO crsql_changes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", change
        )


def get_table_contents(c, table="foo"):
    """Get all rows from a table, sorted by primary key."""
    return c.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()


def assert_tables_equal(c1, c2, table="foo"):
    """Assert two nodes have identical table contents."""
    rows1 = get_table_contents(c1, table)
    rows2 = get_table_contents(c2, table)
    assert rows1 == rows2, f"Tables diverged:\n  node1: {rows1}\n  node2: {rows2}"


def assert_convergence(nodes, table="foo"):
    """Assert all nodes have identical table contents."""
    reference = get_table_contents(nodes[0], table)
    for i, node in enumerate(nodes[1:], 1):
        rows = get_table_contents(node, table)
        assert rows == reference, (
            f"Node {i} diverged from node 0:\n"
            f"  node 0: {reference}\n"
            f"  node {i}: {rows}"
        )


# ===========================================================================
# 1. PARTIAL SYNC — interrupted transfers where only some changes arrive
# ===========================================================================


class TestPartialSync:
    """
    When a phone syncs to a laptop and WiFi drops mid-transfer, only some
    changes arrive. The remaining changes must be deliverable later without
    data loss.
    """

    def test_split_batch_converges(self):
        """
        Apply first half of a changeset, then second half.
        Must produce the same result as applying all at once.
        """
        sender = make_simple_schema()
        receiver_split = make_simple_schema()
        receiver_all = make_simple_schema()

        # Generate changes on sender
        for i in range(1, 11):
            sender.execute("INSERT INTO foo VALUES (?, ?)", (i, i * 10))
        sender.commit()

        changes = get_all_changes(sender)
        mid = len(changes) // 2

        # Split delivery
        apply_changes(receiver_split, changes[:mid])
        apply_changes(receiver_split, changes[mid:])

        # Full delivery
        apply_changes(receiver_all, changes)

        assert_tables_equal(receiver_split, receiver_all)

        close(sender)
        close(receiver_split)
        close(receiver_all)

    def test_out_of_order_db_versions(self):
        """
        Apply changes in reverse db_version order within a batch.
        CRDTs should converge regardless of delivery order.
        """
        sender = make_simple_schema()
        receiver_ordered = make_simple_schema()
        receiver_reversed = make_simple_schema()

        for i in range(1, 6):
            sender.execute("INSERT INTO foo VALUES (?, ?)", (i, i * 10))
            sender.commit()  # Each insert gets its own db_version

        changes = get_all_changes(sender)

        apply_changes(receiver_ordered, changes)
        apply_changes(receiver_reversed, list(reversed(changes)))

        assert_tables_equal(receiver_ordered, receiver_reversed)

        close(sender)
        close(receiver_ordered)
        close(receiver_reversed)

    def test_random_subset_then_fill_gaps(self):
        """
        Apply a random subset of changes, then fill in the gaps.
        Simulates unreliable transport where some packets arrive, then
        a retry fills in the rest.
        """
        sender = make_simple_schema()
        receiver_partial = make_simple_schema()
        receiver_all = make_simple_schema()

        for i in range(1, 21):
            sender.execute("INSERT INTO foo VALUES (?, ?)", (i, i * 10))
        sender.commit()

        changes = get_all_changes(sender)

        # Deliver random 50%
        indices = list(range(len(changes)))
        random.shuffle(indices)
        first_batch_idx = sorted(indices[: len(indices) // 2])
        second_batch_idx = sorted(indices[len(indices) // 2 :])

        apply_changes(receiver_partial, [changes[i] for i in first_batch_idx])
        apply_changes(receiver_partial, [changes[i] for i in second_batch_idx])

        apply_changes(receiver_all, changes)

        assert_tables_equal(receiver_partial, receiver_all)

        close(sender)
        close(receiver_partial)
        close(receiver_all)

    def test_partial_sync_with_updates(self):
        """
        Sender inserts rows then updates them. Partial sync delivers some
        inserts and some updates. The remaining changes fill in later.
        """
        sender = make_simple_schema()
        receiver_split = make_simple_schema()
        receiver_all = make_simple_schema()

        # Insert phase
        for i in range(1, 6):
            sender.execute("INSERT INTO foo VALUES (?, ?)", (i, i * 10))
        sender.commit()

        # Update phase
        for i in range(1, 6):
            sender.execute("UPDATE foo SET b = ? WHERE a = ?", (i * 100, i))
        sender.commit()

        changes = get_all_changes(sender)

        # Deliver every other change (mixes inserts and updates)
        first_batch = changes[::2]
        second_batch = changes[1::2]

        apply_changes(receiver_split, first_batch)
        apply_changes(receiver_split, second_batch)

        apply_changes(receiver_all, changes)

        assert_tables_equal(receiver_split, receiver_all)

        close(sender)
        close(receiver_split)
        close(receiver_all)

    def test_partial_sync_with_deletes(self):
        """
        Sender inserts rows, deletes some, inserts more. Partial sync
        delivers the delete but not the original insert. The insert
        arrives later. Final state should match.
        """
        sender = make_simple_schema()
        receiver_split = make_simple_schema()
        receiver_all = make_simple_schema()

        # Insert
        for i in range(1, 6):
            sender.execute("INSERT INTO foo VALUES (?, ?)", (i, i * 10))
        sender.commit()

        # Delete rows 2 and 4
        sender.execute("DELETE FROM foo WHERE a = 2")
        sender.execute("DELETE FROM foo WHERE a = 4")
        sender.commit()

        changes = get_all_changes(sender)

        # Deliver second half first (contains deletes), then first half (inserts)
        mid = len(changes) // 2
        apply_changes(receiver_split, changes[mid:])
        apply_changes(receiver_split, changes[:mid])

        apply_changes(receiver_all, changes)

        assert_tables_equal(receiver_split, receiver_all)

        close(sender)
        close(receiver_split)
        close(receiver_all)

    def test_rollback_then_reapply(self):
        """
        Apply changes, rollback mid-transaction, then re-apply from scratch.
        Simulates a sync that detects corruption and retries.
        """
        sender = make_simple_schema()
        receiver = make_simple_schema()

        for i in range(1, 6):
            sender.execute("INSERT INTO foo VALUES (?, ?)", (i, i * 10))
        sender.commit()

        changes = get_all_changes(sender)

        # Start applying, then rollback
        apply_changes_no_commit(receiver, changes[:3])
        receiver.rollback()

        # Now apply everything cleanly
        apply_changes(receiver, changes)

        assert_tables_equal(sender, receiver)

        close(sender)
        close(receiver)

    def test_interleaved_local_writes_during_partial_sync(self):
        """
        A node receives partial sync, makes local writes, then receives
        the rest of the sync. Local writes should not be lost.
        """
        sender = make_simple_schema()
        receiver = make_simple_schema()

        for i in range(1, 6):
            sender.execute("INSERT INTO foo VALUES (?, ?)", (i, i * 10))
        sender.commit()

        changes = get_all_changes(sender)
        mid = len(changes) // 2

        # Partial sync
        apply_changes(receiver, changes[:mid])

        # Local write on receiver
        receiver.execute("INSERT INTO foo VALUES (100, 999)")
        receiver.commit()

        # Rest of sync
        apply_changes(receiver, changes[mid:])

        # Receiver should have all sender rows + its own local row
        receiver_rows = get_table_contents(receiver)
        sender_rows = get_table_contents(sender)

        # All sender rows present
        for row in sender_rows:
            assert row in receiver_rows, f"Sender row {row} missing from receiver"

        # Local row preserved
        assert (100, 999) in receiver_rows, "Local write lost during partial sync"

        close(sender)
        close(receiver)


# ===========================================================================
# 2. DUPLICATE DELIVERY — at-least-once transport applies changes twice
# ===========================================================================


class TestDuplicateDelivery:
    """
    At-least-once delivery means the same change may arrive multiple times.
    Every duplicate application must be idempotent — the result should be
    identical to applying the change once.
    """

    def test_immediate_duplicate_inserts(self):
        """Apply every change twice in immediate succession."""
        sender = make_simple_schema()
        receiver_once = make_simple_schema()
        receiver_twice = make_simple_schema()

        for i in range(1, 11):
            sender.execute("INSERT INTO foo VALUES (?, ?)", (i, i * 10))
        sender.commit()

        changes = get_all_changes(sender)

        apply_changes(receiver_once, changes)
        apply_changes(receiver_twice, changes + changes)

        assert_tables_equal(receiver_once, receiver_twice)

        close(sender)
        close(receiver_once)
        close(receiver_twice)

    def test_duplicate_after_local_edits(self):
        """
        Apply a batch, make local edits, re-apply the same batch.
        Local edits should win if they have higher col_version.
        """
        sender = make_simple_schema()
        receiver = make_simple_schema()
        reference = make_simple_schema()

        # Sender writes
        for i in range(1, 6):
            sender.execute("INSERT INTO foo VALUES (?, ?)", (i, i * 10))
        sender.commit()

        changes = get_all_changes(sender)

        # First application
        apply_changes(receiver, changes)
        apply_changes(reference, changes)

        # Local edits on receiver (higher col_version for these rows)
        receiver.execute("UPDATE foo SET b = 999 WHERE a = 1")
        receiver.commit()
        reference.execute("UPDATE foo SET b = 999 WHERE a = 1")
        reference.commit()

        # Duplicate delivery of original changes — should NOT overwrite local edit
        apply_changes(receiver, changes)

        assert_tables_equal(receiver, reference)

        close(sender)
        close(receiver)
        close(reference)

    def test_duplicate_with_new_changes(self):
        """
        Apply a batch, then re-apply the batch mixed with additional new changes.
        Both old (duplicate) and new changes should be handled correctly.
        """
        sender = make_simple_schema()
        receiver = make_simple_schema()
        reference = make_simple_schema()

        # First batch
        for i in range(1, 6):
            sender.execute("INSERT INTO foo VALUES (?, ?)", (i, i * 10))
        sender.commit()

        first_changes = get_all_changes(sender)

        apply_changes(receiver, first_changes)
        apply_changes(reference, first_changes)

        # More changes on sender
        for i in range(6, 11):
            sender.execute("INSERT INTO foo VALUES (?, ?)", (i, i * 10))
        sender.commit()

        all_changes = get_all_changes(sender)

        # Receiver gets ALL changes (including duplicates of the first batch)
        apply_changes(receiver, all_changes)
        # Reference gets only the new changes
        new_changes = get_all_changes(sender, since=1)
        apply_changes(reference, new_changes)

        assert_tables_equal(receiver, reference)

        close(sender)
        close(receiver)
        close(reference)

    def test_duplicate_deletes(self):
        """
        Apply a delete, then apply the same delete again.
        The row should remain deleted with no side effects.
        """
        sender = make_simple_schema()
        receiver = make_simple_schema()

        sender.execute("INSERT INTO foo VALUES (1, 10)")
        sender.commit()

        # Sync the insert
        insert_changes = get_all_changes(sender)
        apply_changes(receiver, insert_changes)

        # Delete on sender
        sender.execute("DELETE FROM foo WHERE a = 1")
        sender.commit()

        delete_changes = get_all_changes(sender, since=1)

        # Apply delete twice
        apply_changes(receiver, delete_changes)
        apply_changes(receiver, delete_changes)

        # Row should be deleted
        rows = get_table_contents(receiver)
        assert len(rows) == 0, f"Expected empty table, got: {rows}"

        close(sender)
        close(receiver)

    def test_duplicate_resurrection(self):
        """
        Delete a row, then re-insert it (resurrection). Apply the resurrection
        changes twice. The row should exist exactly once.
        """
        sender = make_simple_schema()
        receiver = make_simple_schema()

        # Insert, sync, delete, resurrect
        sender.execute("INSERT INTO foo VALUES (1, 10)")
        sender.commit()

        insert_changes = get_all_changes(sender)
        apply_changes(receiver, insert_changes)

        sender.execute("DELETE FROM foo WHERE a = 1")
        sender.commit()
        delete_changes = get_all_changes(sender, since=1)
        apply_changes(receiver, delete_changes)

        sender.execute("INSERT INTO foo VALUES (1, 20)")
        sender.commit()
        resurrect_changes = get_all_changes(sender, since=2)

        # Apply resurrection twice
        apply_changes(receiver, resurrect_changes)
        apply_changes(receiver, resurrect_changes)

        rows = get_table_contents(receiver)
        assert rows == [(1, 20)], f"Expected [(1, 20)], got: {rows}"

        close(sender)
        close(receiver)

    def test_full_idempotent_sync_cycle(self):
        """
        Two nodes make concurrent changes, sync both ways, then sync again.
        The second sync should be a no-op and both nodes should be converged.
        """
        a = make_simple_schema()
        b = make_simple_schema()

        # Concurrent writes
        a.execute("INSERT INTO foo VALUES (1, 100)")
        a.commit()
        b.execute("INSERT INTO foo VALUES (2, 200)")
        b.commit()

        # First sync cycle: A -> B, B -> A
        a_changes = get_all_changes(a)
        b_changes = get_all_changes(b)
        apply_changes(b, a_changes)
        apply_changes(a, b_changes)

        snapshot_a = get_table_contents(a)
        snapshot_b = get_table_contents(b)
        assert_tables_equal(a, b)

        # Second sync cycle (should be no-op)
        a_changes_2 = get_all_changes(a)
        b_changes_2 = get_all_changes(b)
        apply_changes(b, a_changes_2)
        apply_changes(a, b_changes_2)

        assert get_table_contents(a) == snapshot_a, "Second sync changed node A"
        assert get_table_contents(b) == snapshot_b, "Second sync changed node B"
        assert_tables_equal(a, b)

        close(a)
        close(b)


# ===========================================================================
# 3. OUT-OF-ORDER DELIVERY — changes arrive in shuffled order
# ===========================================================================


class TestOutOfOrderDelivery:
    """
    On unreliable networks, changes may arrive in any order. The CRDT must
    converge to the same state regardless of delivery order.
    """

    def test_shuffled_delivery_converges(self):
        """
        Apply the same set of changes in 5 different random orders.
        All receivers must converge to the same state.
        """
        sender = make_multi_column_schema()
        receivers = [make_multi_column_schema() for _ in range(5)]

        # Generate a variety of operations
        sender.execute("INSERT INTO items VALUES (1, 'alpha', 10, 'active')")
        sender.execute("INSERT INTO items VALUES (2, 'beta', 20, 'active')")
        sender.execute("INSERT INTO items VALUES (3, 'gamma', 30, 'active')")
        sender.commit()
        sender.execute("UPDATE items SET value = 100 WHERE id = 1")
        sender.execute("DELETE FROM items WHERE id = 2")
        sender.commit()
        sender.execute("INSERT INTO items VALUES (2, 'beta-v2', 25, 'revived')")
        sender.execute("UPDATE items SET status = 'archived' WHERE id = 3")
        sender.commit()

        changes = get_all_changes(sender)

        for receiver in receivers:
            shuffled = list(changes)
            random.shuffle(shuffled)
            apply_changes(receiver, shuffled)

        assert_convergence(receivers, table="items")

        # Also verify they match the sender
        for receiver in receivers:
            assert_tables_equal(sender, receiver, table="items")

        close(sender)
        for r in receivers:
            close(r)

    def test_causal_inversion_insert_then_update(self):
        """
        An update arrives before the insert it depends on.
        The CRDT should handle this gracefully.
        """
        sender = make_simple_schema()
        receiver = make_simple_schema()
        reference = make_simple_schema()

        sender.execute("INSERT INTO foo VALUES (1, 10)")
        sender.commit()
        sender.execute("UPDATE foo SET b = 20 WHERE a = 1")
        sender.commit()

        changes = get_all_changes(sender)

        # Apply update before insert (reversed order)
        apply_changes(reference, changes)
        apply_changes(receiver, list(reversed(changes)))

        assert_tables_equal(receiver, reference)

        close(sender)
        close(receiver)
        close(reference)

    def test_causal_inversion_insert_then_delete(self):
        """
        A delete arrives before the insert. The row should end up deleted.
        """
        sender = make_simple_schema()
        receiver = make_simple_schema()
        reference = make_simple_schema()

        sender.execute("INSERT INTO foo VALUES (1, 10)")
        sender.commit()
        sender.execute("DELETE FROM foo WHERE a = 1")
        sender.commit()

        changes = get_all_changes(sender)

        apply_changes(reference, changes)
        apply_changes(receiver, list(reversed(changes)))

        assert_tables_equal(receiver, reference)

        close(sender)
        close(receiver)
        close(reference)


# ===========================================================================
# 4. MULTI-NODE PARTIAL SYNC — realistic local-first topology
# ===========================================================================


class TestMultiNodePartialSync:
    """
    Local-first apps typically sync through a relay or peer-to-peer.
    Changes from many nodes arrive at different times in different orders.
    """

    def test_three_node_partial_mesh(self):
        """
        Three nodes make changes. Each pair syncs partially, then completes.
        All should converge.
        """
        a = make_simple_schema()
        b = make_simple_schema()
        c = make_simple_schema()

        # Each node makes unique writes
        a.execute("INSERT INTO foo VALUES (1, 100)")
        a.execute("INSERT INTO foo VALUES (2, 200)")
        a.commit()

        b.execute("INSERT INTO foo VALUES (3, 300)")
        b.execute("INSERT INTO foo VALUES (4, 400)")
        b.commit()

        c.execute("INSERT INTO foo VALUES (5, 500)")
        c.execute("INSERT INTO foo VALUES (6, 600)")
        c.commit()

        a_changes = get_all_changes(a)
        b_changes = get_all_changes(b)
        c_changes = get_all_changes(c)

        # Partial sync: A sends half to B, B sends half to C
        apply_changes(b, a_changes[: len(a_changes) // 2])
        apply_changes(c, b_changes[: len(b_changes) // 2])

        # Complete sync: all remaining changes flow everywhere
        # A gets everything from B and C
        apply_changes(a, get_all_changes(b))
        apply_changes(a, get_all_changes(c))

        # B gets everything from A and C
        apply_changes(b, get_all_changes(a))
        apply_changes(b, get_all_changes(c))

        # C gets everything from A and B
        apply_changes(c, get_all_changes(a))
        apply_changes(c, get_all_changes(b))

        assert_convergence([a, b, c])

        close(a)
        close(b)
        close(c)

    def test_conflict_resolution_with_partial_sync(self):
        """
        Two nodes update the same row concurrently. One node receives a
        partial sync (only one column of the update), then the rest later.
        The final state should match full sync.
        """
        a = make_multi_column_schema()
        b = make_multi_column_schema()
        receiver_partial = make_multi_column_schema()
        receiver_full = make_multi_column_schema()

        # Both start with the same row
        a.execute("INSERT INTO items VALUES (1, 'original', 0, 'new')")
        a.commit()
        a_insert = get_all_changes(a)
        apply_changes(b, a_insert)
        apply_changes(receiver_partial, a_insert)
        apply_changes(receiver_full, a_insert)

        # Concurrent updates to different columns
        a.execute("UPDATE items SET name = 'from-a', value = 100 WHERE id = 1")
        a.commit()
        b.execute("UPDATE items SET status = 'from-b' WHERE id = 1")
        b.commit()

        a_updates = get_all_changes(a, since=1)
        b_updates = get_all_changes(b, since=1)
        all_updates = a_updates + b_updates

        # Full delivery
        apply_changes(receiver_full, all_updates)

        # Partial: first just A's changes, then B's
        apply_changes(receiver_partial, a_updates[:1])  # partial A
        apply_changes(receiver_partial, b_updates)  # all B
        apply_changes(receiver_partial, a_updates[1:])  # rest of A

        assert_tables_equal(receiver_partial, receiver_full, table="items")

        close(a)
        close(b)
        close(receiver_partial)
        close(receiver_full)

    def test_five_node_star_topology(self):
        """
        Five edge nodes sync through a central hub. Some edges are offline
        during parts of the sync. All should eventually converge.
        """
        hub = make_simple_schema()
        edges = [make_simple_schema() for _ in range(5)]

        # Each edge makes local writes
        for i, edge in enumerate(edges):
            base = (i + 1) * 100
            edge.execute("INSERT INTO foo VALUES (?, ?)", (base + 1, base + 1))
            edge.execute("INSERT INTO foo VALUES (?, ?)", (base + 2, base + 2))
            edge.commit()

        # Phase 1: edges 0, 1, 2 sync to hub (3 and 4 are "offline")
        for edge in edges[:3]:
            apply_changes(hub, get_all_changes(edge))

        # Phase 2: hub syncs current state to edges 0 and 1
        hub_changes = get_all_changes(hub)
        apply_changes(edges[0], hub_changes)
        apply_changes(edges[1], hub_changes)

        # Phase 3: edges 3 and 4 come online, sync to hub
        for edge in edges[3:]:
            apply_changes(hub, get_all_changes(edge))

        # Phase 4: hub syncs full state to all edges
        hub_all = get_all_changes(hub)
        for edge in edges:
            apply_changes(edge, hub_all)

        assert_convergence([hub] + edges)

        close(hub)
        for e in edges:
            close(e)


# ===========================================================================
# 5. PROPERTY-BASED TESTS — randomized scenarios
# ===========================================================================


class TestPropertyBased:
    """
    Hypothesis-driven tests that generate random operations and delivery
    orders to find edge cases.
    """

    @given(data=data())
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_any_delivery_order_converges(self, data):
        """
        Generate random inserts/updates/deletes, collect changes, deliver
        in a random permutation. Assert convergence with in-order delivery.
        """
        sender = make_simple_schema()
        receiver_ordered = make_simple_schema()
        receiver_shuffled = make_simple_schema()

        num_ops = data.draw(integers(min_value=1, max_value=20))

        existing_keys = []
        for _ in range(num_ops):
            op = data.draw(integers(min_value=0, max_value=2))
            if op == 0 or not existing_keys:
                # Insert
                key = data.draw(integers(min_value=1, max_value=1000))
                val = data.draw(integers(min_value=0, max_value=10000))
                sender.execute(
                    "INSERT OR REPLACE INTO foo VALUES (?, ?)", (key, val)
                )
                if key not in existing_keys:
                    existing_keys.append(key)
            elif op == 1 and existing_keys:
                # Update
                key = random.choice(existing_keys)
                val = data.draw(integers(min_value=0, max_value=10000))
                sender.execute("UPDATE foo SET b = ? WHERE a = ?", (val, key))
            elif op == 2 and existing_keys:
                # Delete
                key = random.choice(existing_keys)
                sender.execute("DELETE FROM foo WHERE a = ?", (key,))
                existing_keys.remove(key)
            sender.commit()

        changes = get_all_changes(sender)
        if not changes:
            close(sender)
            close(receiver_ordered)
            close(receiver_shuffled)
            return

        # Ordered delivery
        apply_changes(receiver_ordered, changes)

        # Shuffled delivery
        shuffled = list(changes)
        random.shuffle(shuffled)
        apply_changes(receiver_shuffled, shuffled)

        assert_tables_equal(receiver_ordered, receiver_shuffled)

        close(sender)
        close(receiver_ordered)
        close(receiver_shuffled)

    @given(data=data())
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_duplicate_delivery_is_idempotent(self, data):
        """
        Generate random changes, apply them, then apply random subsets again.
        State should be identical to single application.
        """
        sender = make_simple_schema()
        receiver_once = make_simple_schema()
        receiver_duped = make_simple_schema()

        num_ops = data.draw(integers(min_value=1, max_value=15))

        for _ in range(num_ops):
            key = data.draw(integers(min_value=1, max_value=100))
            val = data.draw(integers(min_value=0, max_value=10000))
            sender.execute("INSERT OR REPLACE INTO foo VALUES (?, ?)", (key, val))
        sender.commit()

        changes = get_all_changes(sender)

        # Single delivery
        apply_changes(receiver_once, changes)

        # Deliver all, then re-deliver a random subset
        apply_changes(receiver_duped, changes)
        num_dupes = data.draw(integers(min_value=1, max_value=len(changes)))
        dupe_indices = random.sample(range(len(changes)), num_dupes)
        dupes = [changes[i] for i in dupe_indices]
        apply_changes(receiver_duped, dupes)

        assert_tables_equal(receiver_once, receiver_duped)

        close(sender)
        close(receiver_once)
        close(receiver_duped)
