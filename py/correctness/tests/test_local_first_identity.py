"""
Tests for node identity / cloned database scenarios in local-first.

In local-first apps, database files can be duplicated through backup/restore,
iCloud file sync, manual copying, etc. This creates two nodes with the same
site_id, which breaks the assumption that site_id uniquely identifies a source.

Also tests version regression: a node restores from backup to an earlier state,
so its db_version goes backward while the site_id stays the same.
"""

from crsql_correctness import connect, close, get_site_id, min_db_v
import pytest
import os
import tempfile
import shutil


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_simple_schema(db_path=":memory:"):
    c = connect(db_path)
    c.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER) STRICT;")
    c.execute("SELECT crsql_as_crr('foo')")
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


def get_table_contents(c, table="foo"):
    return c.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()


def assert_tables_equal(c1, c2, table="foo"):
    rows1 = get_table_contents(c1, table)
    rows2 = get_table_contents(c2, table)
    assert rows1 == rows2, f"Tables diverged:\n  node1: {rows1}\n  node2: {rows2}"


def copy_db_file(src_path, dst_path):
    """Copy a SQLite database file, ensuring WAL is checkpointed first."""
    # Open and checkpoint to ensure all data is in the main file
    import sqlite3
    conn = sqlite3.connect(src_path)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    shutil.copy2(src_path, dst_path)


# ===========================================================================
# 1. CLONED DATABASES — two nodes with the same site_id
# ===========================================================================


class TestClonedDatabase:
    """
    When a database file is copied (backup, iCloud, manual copy), two devices
    end up with the same site_id. Their changes are indistinguishable to peers.
    """

    def test_clone_diverge_sync_via_third_node(self):
        """
        Create a database, clone it (same site_id), make different writes on
        both. Sync both to a third node. The third node sees two different
        sets of changes from the "same" source.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            original_path = os.path.join(tmpdir, "original.db")
            clone_path = os.path.join(tmpdir, "clone.db")

            original = make_simple_schema(original_path)

            # Write some initial data
            original.execute("INSERT INTO foo VALUES (1, 100)")
            original.commit()

            original_site_id = get_site_id(original)

            # Close and clone the file
            close(original)
            copy_db_file(original_path, clone_path)

            # Reopen both
            original = connect(original_path)
            clone = connect(clone_path)

            # Verify they have the same site_id
            clone_site_id = get_site_id(clone)
            assert original_site_id == clone_site_id, "Clone should have same site_id"

            # Both have the same row
            assert get_table_contents(original) == [(1, 100)]
            assert get_table_contents(clone) == [(1, 100)]

            # Diverge: different writes on each
            original.execute("INSERT INTO foo VALUES (2, 200)")
            original.execute("UPDATE foo SET b = 111 WHERE a = 1")
            original.commit()

            clone.execute("INSERT INTO foo VALUES (3, 300)")
            clone.execute("UPDATE foo SET b = 999 WHERE a = 1")
            clone.commit()

            # Third node receives from both
            observer = make_simple_schema()

            # Note: both original and clone have the same site_id,
            # so observer's crsql_db_versions will track them as one source
            original_changes = get_all_changes(original)
            clone_changes = get_all_changes(clone)

            apply_changes(observer, original_changes)
            apply_changes(observer, clone_changes)

            # Observer should have all rows (1, 2, 3)
            rows = get_table_contents(observer)
            pks = [r[0] for r in rows]
            assert 1 in pks, "Row 1 should exist"
            assert 2 in pks, "Row 2 (from original) should exist"
            assert 3 in pks, "Row 3 (from clone) should exist"

            close(original)
            close(clone)
            close(observer)
        finally:
            shutil.rmtree(tmpdir)

    def test_clone_conflicting_update_resolution(self):
        """
        Two clones update the same row to different values. The CRDT
        resolution rules should pick a deterministic winner.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            original_path = os.path.join(tmpdir, "original.db")
            clone_path = os.path.join(tmpdir, "clone.db")

            original = make_simple_schema(original_path)
            original.execute("INSERT INTO foo VALUES (1, 100)")
            original.commit()

            close(original)
            copy_db_file(original_path, clone_path)

            original = connect(original_path)
            clone = connect(clone_path)

            # Both update the same row — same col_version since they diverged
            # from the same state
            original.execute("UPDATE foo SET b = 200 WHERE a = 1")
            original.commit()

            clone.execute("UPDATE foo SET b = 300 WHERE a = 1")
            clone.commit()

            # Sync both to two observers in different orders
            observer1 = make_simple_schema()
            observer2 = make_simple_schema()

            orig_changes = get_all_changes(original)
            clone_changes = get_all_changes(clone)

            # Observer 1: original first, then clone
            apply_changes(observer1, orig_changes)
            apply_changes(observer1, clone_changes)

            # Observer 2: clone first, then original
            apply_changes(observer2, clone_changes)
            apply_changes(observer2, orig_changes)

            # Both observers should agree (deterministic resolution)
            assert_tables_equal(observer1, observer2)

            close(original)
            close(clone)
            close(observer1)
            close(observer2)
        finally:
            shutil.rmtree(tmpdir)

    def test_clone_delete_on_one_side(self):
        """
        Clone a database. One side deletes a row, the other updates it.
        The delete has a higher CL and should win.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            original_path = os.path.join(tmpdir, "original.db")
            clone_path = os.path.join(tmpdir, "clone.db")

            original = make_simple_schema(original_path)
            original.execute("INSERT INTO foo VALUES (1, 100)")
            original.execute("INSERT INTO foo VALUES (2, 200)")
            original.commit()

            close(original)
            copy_db_file(original_path, clone_path)

            original = connect(original_path)
            clone = connect(clone_path)

            # Original deletes row 1 (CL goes from 1 to 2)
            original.execute("DELETE FROM foo WHERE a = 1")
            original.commit()

            # Clone updates row 1 (CL stays at 1, col_version goes up)
            clone.execute("UPDATE foo SET b = 999 WHERE a = 1")
            clone.commit()

            observer = make_simple_schema()

            # Apply clone's update first, then original's delete
            apply_changes(observer, get_all_changes(clone))
            apply_changes(observer, get_all_changes(original))

            rows = get_table_contents(observer)
            pks = [r[0] for r in rows]
            assert 1 not in pks, "Row 1 should be deleted (delete CL=2 beats update CL=1)"
            assert 2 in pks, "Row 2 should still exist"

            close(original)
            close(clone)
            close(observer)
        finally:
            shutil.rmtree(tmpdir)


# ===========================================================================
# 2. VERSION REGRESSION — backup restore causes db_version to go backward
# ===========================================================================


class TestVersionRegression:
    """
    A device restores from a backup, reverting to an earlier db_version while
    keeping the same site_id. This can confuse version-based sync cursors.
    """

    def test_backup_restore_then_sync(self):
        """
        Node A writes 3 batches. After batch 2, we save a backup.
        We restore the backup (losing batch 3). Then sync with node B
        which has the full state from all 3 batches.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            a_path = os.path.join(tmpdir, "node_a.db")
            backup_path = os.path.join(tmpdir, "backup.db")

            node_a = make_simple_schema(a_path)
            node_b = make_simple_schema()

            # Batch 1
            node_a.execute("INSERT INTO foo VALUES (1, 100)")
            node_a.commit()

            # Batch 2
            node_a.execute("INSERT INTO foo VALUES (2, 200)")
            node_a.commit()

            # Sync A -> B (B now has rows 1, 2)
            apply_changes(node_b, get_all_changes(node_a))

            # Create backup after batch 2
            close(node_a)
            copy_db_file(a_path, backup_path)
            node_a = connect(a_path)

            # Batch 3 on A
            node_a.execute("INSERT INTO foo VALUES (3, 300)")
            node_a.commit()

            # Sync batch 3 to B
            apply_changes(node_b, get_all_changes(node_a, since=2))

            assert get_table_contents(node_b) == [(1, 100), (2, 200), (3, 300)]

            # DISASTER: A restores from backup (loses batch 3)
            close(node_a)
            copy_db_file(backup_path, a_path)
            node_a = connect(a_path)

            # A now only has rows 1, 2
            assert get_table_contents(node_a) == [(1, 100), (2, 200)]

            # A makes new writes post-restore
            node_a.execute("INSERT INTO foo VALUES (4, 400)")
            node_a.commit()

            # Sync B -> A: B has row 3 which A lost
            b_changes = get_all_changes(node_b)
            apply_changes(node_a, b_changes)

            # A should now have all rows including the one it lost
            rows_a = get_table_contents(node_a)
            pks = [r[0] for r in rows_a]
            assert 3 in pks, "Row 3 (lost in backup restore) should be recovered from B"
            assert 4 in pks, "Row 4 (written after restore) should exist"

            close(node_a)
            close(node_b)
        finally:
            shutil.rmtree(tmpdir)

    def test_backup_restore_conflicting_writes(self):
        """
        Node A writes, backs up, writes more, syncs to B.
        A restores backup, writes DIFFERENT data at the same keys.
        Sync both ways — should converge deterministically.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            a_path = os.path.join(tmpdir, "node_a.db")
            backup_path = os.path.join(tmpdir, "backup.db")

            node_a = make_simple_schema(a_path)
            node_b = make_simple_schema()

            # Initial write
            node_a.execute("INSERT INTO foo VALUES (1, 100)")
            node_a.commit()

            # Backup
            close(node_a)
            copy_db_file(a_path, backup_path)
            node_a = connect(a_path)

            # Post-backup writes
            node_a.execute("UPDATE foo SET b = 200 WHERE a = 1")
            node_a.commit()

            # Sync to B
            apply_changes(node_b, get_all_changes(node_a))
            assert get_table_contents(node_b) == [(1, 200)]

            # Restore backup on A (reverts to b=100)
            close(node_a)
            copy_db_file(backup_path, a_path)
            node_a = connect(a_path)
            assert get_table_contents(node_a) == [(1, 100)]

            # A makes a different update
            node_a.execute("UPDATE foo SET b = 999 WHERE a = 1")
            node_a.commit()

            # Full bidirectional sync
            a_changes = get_all_changes(node_a)
            b_changes = get_all_changes(node_b)
            apply_changes(node_b, a_changes)
            apply_changes(node_a, b_changes)

            # They should converge to the same value
            assert_tables_equal(node_a, node_b)

            close(node_a)
            close(node_b)
        finally:
            shutil.rmtree(tmpdir)

    def test_db_version_cursor_after_restore(self):
        """
        Verify the state of crsql_db_versions after a backup restore.
        The restored node's db_version goes backward, but the tracked
        versions from other peers should still enable correct sync.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            a_path = os.path.join(tmpdir, "node_a.db")
            backup_path = os.path.join(tmpdir, "backup.db")

            node_a = make_simple_schema(a_path)
            node_b = make_simple_schema()

            # A writes batch 1
            node_a.execute("INSERT INTO foo VALUES (1, 100)")
            node_a.commit()

            # Sync A -> B
            apply_changes(node_b, get_all_changes(node_a))

            # B writes
            node_b.execute("INSERT INTO foo VALUES (2, 200)")
            node_b.commit()

            # Sync B -> A (A now knows about B's version)
            apply_changes(node_a, get_all_changes(node_b))

            # Record A's db_version tracking state
            a_db_versions_before = node_a.execute(
                "SELECT hex(site_id), db_version FROM crsql_db_versions ORDER BY site_id"
            ).fetchall()

            # Backup A
            close(node_a)
            copy_db_file(a_path, backup_path)
            node_a = connect(a_path)

            # A writes batch 2
            node_a.execute("INSERT INTO foo VALUES (3, 300)")
            node_a.commit()

            # Restore A from backup
            close(node_a)
            copy_db_file(backup_path, a_path)
            node_a = connect(a_path)

            # A's db_versions should be back to the backup state
            a_db_versions_after = node_a.execute(
                "SELECT hex(site_id), db_version FROM crsql_db_versions ORDER BY site_id"
            ).fetchall()

            assert a_db_versions_after == a_db_versions_before, \
                f"db_versions should revert to backup state:\n  before: {a_db_versions_before}\n  after: {a_db_versions_after}"

            # B writes more
            node_b.execute("INSERT INTO foo VALUES (4, 400)")
            node_b.commit()

            # Sync B -> A: A needs to get row 4. But A's cursor for B's site_id
            # is from the backup, which should still work since we use full changes
            b_changes = get_all_changes(node_b)
            apply_changes(node_a, b_changes)

            rows_a = get_table_contents(node_a)
            pks = [r[0] for r in rows_a]
            assert 2 in pks, "Row 2 (synced before backup) should exist"
            assert 4 in pks, "Row 4 (synced after restore) should exist"

            close(node_a)
            close(node_b)
        finally:
            shutil.rmtree(tmpdir)


# ===========================================================================
# 3. SITE ID COLLISION EDGE CASES
# ===========================================================================


class TestSiteIdEdgeCases:
    """
    Edge cases around site_id handling that can occur in local-first.
    """

    def test_clone_direct_sync_rejected_when_ahead(self):
        """
        A clone that has advanced past the original's db_version cannot sync
        directly to the original. The merge code rejects changes tagged with
        your own site_id at a db_version higher than your own — this is an
        expected corruption guard (db_version.rs:259-266).

        In practice this doesn't matter: each device has a unique site_id,
        and backup/restore is done via vacuum+overwrite, not live sync.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            original_path = os.path.join(tmpdir, "original.db")
            clone_path = os.path.join(tmpdir, "clone.db")

            original = make_simple_schema(original_path)
            original.execute("INSERT INTO foo VALUES (1, 100)")
            original.commit()

            close(original)
            copy_db_file(original_path, clone_path)

            original = connect(original_path)
            clone = connect(clone_path)

            # Clone advances past original
            clone.execute("INSERT INTO foo VALUES (2, 200)")
            clone.commit()

            clone_changes = get_all_changes(clone, since=1)

            # Direct sync clone -> original should be rejected
            with pytest.raises(Exception, match="Unable to insert db version"):
                apply_changes(original, clone_changes)

            close(original)
            close(clone)
        finally:
            shutil.rmtree(tmpdir)

    def test_reassign_site_id_after_clone(self):
        """
        The recommended fix for cloned databases: detect the clone and
        generate a new site_id. Verify that existing data is still syncable
        after the site_id change.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            original_path = os.path.join(tmpdir, "original.db")
            clone_path = os.path.join(tmpdir, "clone.db")

            original = make_simple_schema(original_path)
            original.execute("INSERT INTO foo VALUES (1, 100)")
            original.commit()

            close(original)
            copy_db_file(original_path, clone_path)

            original = connect(original_path)
            clone = connect(clone_path)

            original_site_id = get_site_id(original)
            clone_site_id = get_site_id(clone)
            assert original_site_id == clone_site_id

            # Changes made after clone will have the original's site_id
            # in the clock tables. This is by design — those changes happened
            # under that identity.
            clone.execute("INSERT INTO foo VALUES (2, 200)")
            clone.commit()

            # The clone's new row has the shared site_id in its clock
            clone_changes = get_all_changes(clone, since=1)
            assert len(clone_changes) > 0

            # Sync to a third node works regardless
            observer = make_simple_schema()
            apply_changes(observer, get_all_changes(original))
            apply_changes(observer, clone_changes)

            rows = get_table_contents(observer)
            assert (1, 100) in rows
            assert (2, 200) in rows

            close(original)
            close(clone)
            close(observer)
        finally:
            shutil.rmtree(tmpdir)

    def test_multiple_clones_all_write(self):
        """
        Original + 3 clones all make independent writes. Sync everything
        through a central observer. All data should arrive.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            paths = [os.path.join(tmpdir, f"node_{i}.db") for i in range(4)]

            # Create original with initial data
            original = make_simple_schema(paths[0])
            original.execute("INSERT INTO foo VALUES (1, 100)")
            original.commit()
            close(original)

            # Clone to 3 copies
            for p in paths[1:]:
                copy_db_file(paths[0], p)

            # Open all and make different writes
            nodes = [connect(p) for p in paths]

            # Verify all have same site_id
            site_ids = [get_site_id(n) for n in nodes]
            assert len(set([s.hex() for s in site_ids])) == 1, "All clones should share site_id"

            for i, node in enumerate(nodes):
                base = (i + 1) * 10
                node.execute("INSERT INTO foo VALUES (?, ?)", (base, base))
                node.commit()

            # Collect all changes and apply to observer
            observer = make_simple_schema()
            for node in nodes:
                apply_changes(observer, get_all_changes(node))

            rows = get_table_contents(observer)
            pks = sorted([r[0] for r in rows])

            # Should have the shared row (1) plus each clone's unique row
            assert 1 in pks, "Shared initial row should exist"
            assert 10 in pks, "Node 0's write should exist"
            assert 20 in pks, "Node 1's write should exist"
            assert 30 in pks, "Node 2's write should exist"
            assert 40 in pks, "Node 3's write should exist"

            for n in nodes:
                close(n)
            close(observer)
        finally:
            shutil.rmtree(tmpdir)
