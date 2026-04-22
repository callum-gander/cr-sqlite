"""
Reproduction tests for known upstream issues (vlcn-io/cr-sqlite).

These tests verify whether bugs reported against the upstream repo are
reproducible in this fork. Each test class corresponds to a GitHub issue.
"""

from crsql_correctness import connect, close, get_site_id, min_db_v
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def get_table_contents(c, table):
    return c.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()


# ===========================================================================
# #433 — DROP TABLE on a CRR causes corruption
# https://github.com/vlcn-io/cr-sqlite/issues/433
# ===========================================================================


class TestIssue433DropTableCorruption:
    """
    Once a table is a CRR, dropping it leaves orphaned clock and PKs tables.
    crsql_changes becomes unqueryable afterward.
    """

    def test_drop_crr_then_query_changes(self):
        """
        Create a CRR, insert data, drop the table, query crsql_changes.
        CONFIRMED BUG: returns 'query aborted' due to orphaned clock table.
        """
        c = connect(":memory:")
        c.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER) STRICT;")
        c.execute("SELECT crsql_as_crr('foo')")
        c.commit()

        c.execute("INSERT INTO foo VALUES (1, 10)")
        c.commit()

        # Verify changes exist before drop
        changes_before = get_all_changes(c)
        assert len(changes_before) > 0

        # Drop the table
        c.execute("DROP TABLE foo")
        c.commit()

        # CONFIRMED: crsql_changes is broken after DROP TABLE
        with pytest.raises(Exception, match="query aborted"):
            c.execute("SELECT * FROM crsql_changes").fetchall()

        close(c)

    def test_drop_crr_leaves_orphaned_tables(self):
        """
        After dropping a CRR table, the __crsql_clock and __crsql_pks
        tables should ideally be cleaned up. Check if they're orphaned.
        """
        c = connect(":memory:")
        c.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER) STRICT;")
        c.execute("SELECT crsql_as_crr('foo')")
        c.commit()

        c.execute("INSERT INTO foo VALUES (1, 10)")
        c.commit()

        # Verify internal tables exist
        tables_before = set(
            r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'foo__crsql_%'"
            ).fetchall()
        )
        assert "foo__crsql_clock" in tables_before
        assert "foo__crsql_pks" in tables_before

        c.execute("DROP TABLE foo")
        c.commit()

        # Check if internal tables are orphaned
        tables_after = set(
            r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'foo__crsql_%'"
            ).fetchall()
        )

        if tables_after:
            print(f"  Orphaned tables after DROP: {tables_after}")
        else:
            print("  Internal tables were cleaned up (not orphaned)")

        # Document what we find — don't assert either way, this is a reproduction test

    def test_drop_crr_via_begin_alter(self):
        """
        begin_alter + DROP + commit_alter should clean up internal tables
        and leave crsql_changes working.
        """
        c = connect(":memory:")
        c.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER) STRICT;")
        c.execute("SELECT crsql_as_crr('foo')")
        c.commit()

        c.execute("INSERT INTO foo VALUES (1, 10)")
        c.commit()

        # Drop via the proper alter path
        c.execute("SELECT crsql_begin_alter('foo')")
        c.execute("DROP TABLE foo")
        c.execute("SELECT crsql_commit_alter('foo')")
        c.commit()

        # Internal tables should be cleaned up
        orphaned = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'foo__crsql_%'"
        ).fetchall()
        assert len(orphaned) == 0, f"Orphaned tables remain: {orphaned}"

        # crsql_changes should still work (returns empty since the only CRR is gone)
        changes = c.execute("SELECT * FROM crsql_changes").fetchall()
        assert len(changes) == 0

        close(c)

    def test_drop_and_recreate_then_sync(self):
        """
        Drop a CRR, recreate it with the same name, then try to sync
        old changes. Upstream reports 'could not find row to merge with'.
        """
        sender = connect(":memory:")
        sender.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER) STRICT;")
        sender.execute("SELECT crsql_as_crr('foo')")
        sender.commit()
        sender.execute("INSERT INTO foo VALUES (1, 10)")
        sender.commit()

        old_changes = get_all_changes(sender)

        receiver = connect(":memory:")
        receiver.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER) STRICT;")
        receiver.execute("SELECT crsql_as_crr('foo')")
        receiver.commit()

        # Sync old data
        apply_changes(receiver, old_changes)
        assert get_table_contents(receiver, "foo") == [(1, 10)]

        # Sender drops and recreates
        sender.execute("DROP TABLE foo")
        sender.commit()
        sender.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER) STRICT;")
        sender.execute("SELECT crsql_as_crr('foo')")
        sender.commit()
        sender.execute("INSERT INTO foo VALUES (2, 20)")
        sender.commit()

        new_changes = get_all_changes(sender)

        # Try to apply new changes to receiver — receiver still has old clock data
        try:
            apply_changes(receiver, new_changes)
            rows = get_table_contents(receiver, "foo")
            print(f"  Drop + recreate + sync: SUCCEEDED — receiver has {rows}")
        except Exception as e:
            print(f"  Drop + recreate + sync: FAILED — {e}")
            receiver.rollback()

        close(sender)
        close(receiver)

    def test_drop_one_crr_raw_breaks_other_crr(self):
        """
        Two CRR tables. Drop one WITHOUT begin/commit alter.
        CONFIRMED BUG: dropping ANY CRR breaks crsql_changes for ALL tables.
        This is the raw DROP path — no fix applied here.
        """
        c = connect(":memory:")
        c.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER) STRICT;")
        c.execute("SELECT crsql_as_crr('foo')")
        c.execute("CREATE TABLE bar (x INTEGER PRIMARY KEY NOT NULL, y TEXT) STRICT;")
        c.execute("SELECT crsql_as_crr('bar')")
        c.commit()

        c.execute("INSERT INTO foo VALUES (1, 10)")
        c.execute("INSERT INTO bar VALUES (1, 'hello')")
        c.commit()

        c.execute("DROP TABLE foo")
        c.commit()

        # Raw DROP still breaks — no alter wrappers means no cleanup
        with pytest.raises(Exception, match="query aborted"):
            c.execute("SELECT * FROM crsql_changes").fetchall()

    def test_drop_one_crr_via_alter_other_crr_still_works(self):
        """
        Two CRR tables. Drop one via begin/commit alter.
        The other should still be queryable via crsql_changes.
        """
        c = connect(":memory:")
        c.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER) STRICT;")
        c.execute("SELECT crsql_as_crr('foo')")
        c.execute("CREATE TABLE bar (x INTEGER PRIMARY KEY NOT NULL, y TEXT) STRICT;")
        c.execute("SELECT crsql_as_crr('bar')")
        c.commit()

        c.execute("INSERT INTO foo VALUES (1, 10)")
        c.execute("INSERT INTO bar VALUES (1, 'hello')")
        c.commit()

        # Drop foo via the proper alter path
        c.execute("SELECT crsql_begin_alter('foo')")
        c.execute("DROP TABLE foo")
        c.execute("SELECT crsql_commit_alter('foo')")
        c.commit()

        # bar's changes should still be queryable
        changes = c.execute("SELECT * FROM crsql_changes").fetchall()
        bar_changes = [r for r in changes if r[0] == 'bar']
        assert len(bar_changes) > 0, "bar changes should still be visible"

        # foo internal tables should be gone
        orphaned = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'foo__crsql_%'"
        ).fetchall()
        assert len(orphaned) == 0

        close(c)


# ===========================================================================
# #380 — Duplicate seq for the same db_version
# https://github.com/vlcn-io/cr-sqlite/issues/380
# ===========================================================================


class TestIssue380DuplicateSeq:
    """
    When merging changes from multiple source db_versions in a single
    transaction, the resulting changes can have duplicate (db_version, seq)
    pairs.
    """

    def test_merge_different_source_versions_in_one_tx(self):
        """
        Exact reproduction from the issue: two changes from different source
        db_versions merged in a single transaction.
        """
        c = connect(":memory:")
        c.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER) STRICT;")
        c.execute("SELECT crsql_as_crr('foo')")
        c.commit()

        # Create some local data to advance db_version
        c.execute("INSERT INTO foo VALUES (1, 1)")
        c.execute("INSERT INTO foo VALUES (2, 2)")
        c.execute("INSERT INTO foo VALUES (3, 3)")
        c.commit()

        remote_site_id = b'\x9B\x1D\x5B\xD2\x0C\x7A\x47\xC2\x8F\xC4\x82\xB3\x3D\xCF\xC4\x12'

        # Merge two changes from different source db_versions in one transaction
        c.execute("BEGIN")
        c.execute(
            "INSERT INTO crsql_changes VALUES ('foo', X'010904', 'b', 4, 1, 1, ?, 1, 0, '0')",
            (remote_site_id,)
        )
        c.execute(
            "INSERT INTO crsql_changes VALUES ('foo', X'010905', 'b', 5, 1, 2, ?, 1, 0, '0')",
            (remote_site_id,)
        )
        c.commit()

        # Check for duplicate (db_version, seq) pairs
        changes = c.execute(
            "SELECT [table], pk, cid, val, db_version, seq FROM crsql_changes "
            "WHERE val IN (4, 5) ORDER BY db_version, seq"
        ).fetchall()

        print(f"  Changes after merge:")
        seen = set()
        duplicates = []
        for ch in changes:
            db_v, seq = ch[4], ch[5]
            print(f"    val={ch[3]} db_version={db_v} seq={seq}")
            key = (db_v, seq)
            if key in seen:
                duplicates.append(key)
            seen.add(key)

        if duplicates:
            print(f"  DUPLICATE (db_version, seq) pairs found: {duplicates}")
        else:
            print("  No duplicate (db_version, seq) pairs — issue may be fixed")

        close(c)

    def test_seq_uniqueness_within_db_version(self):
        """
        More general test: merge changes from 5 different source db_versions
        in a single transaction. Check that every (db_version, seq) pair in
        the output is unique.
        """
        c = connect(":memory:")
        c.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER) STRICT;")
        c.execute("SELECT crsql_as_crr('foo')")
        c.commit()

        remote_site_id = b'\xAA\xBB\xCC\xDD\xEE\xFF\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99'

        c.execute("BEGIN")
        for i in range(1, 6):
            pk_blob = bytes([0x01, 0x09, i + 10])
            c.execute(
                "INSERT INTO crsql_changes VALUES ('foo', ?, 'b', ?, 1, ?, ?, 1, 0, '0')",
                (pk_blob, i * 100, i, remote_site_id)
            )
        c.commit()

        # Get all changes from the merge
        changes = c.execute(
            "SELECT db_version, seq, val FROM crsql_changes ORDER BY db_version, seq"
        ).fetchall()

        # Check uniqueness
        seen = {}
        duplicates = []
        for db_v, seq, val in changes:
            key = (db_v, seq)
            if key in seen:
                duplicates.append((key, seen[key], val))
            seen[key] = val

        if duplicates:
            print(f"  DUPLICATE (db_version, seq) pairs: {duplicates}")
        else:
            print(f"  All {len(changes)} changes have unique (db_version, seq)")

        close(c)

    def test_seq_ordering_for_sync_consumers(self):
        """
        A sync consumer reads changes ordered by (db_version, seq) to
        reconstruct transactions. If seq has duplicates, the consumer
        can't distinguish ordering within a transaction.
        """
        c = connect(":memory:")
        c.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER) STRICT;")
        c.execute("SELECT crsql_as_crr('foo')")
        c.commit()

        remote1 = b'\x11' * 16
        remote2 = b'\x22' * 16

        # Two remotes send changes that get merged in one transaction
        c.execute("BEGIN")
        c.execute(
            "INSERT INTO crsql_changes VALUES ('foo', X'010901', 'b', 10, 1, 1, ?, 1, 0, '0')",
            (remote1,)
        )
        c.execute(
            "INSERT INTO crsql_changes VALUES ('foo', X'010902', 'b', 20, 1, 1, ?, 1, 0, '0')",
            (remote2,)
        )
        c.execute(
            "INSERT INTO crsql_changes VALUES ('foo', X'010903', 'b', 30, 1, 2, ?, 1, 0, '0')",
            (remote1,)
        )
        c.commit()

        # A sync consumer queries changes
        changes = c.execute(
            "SELECT db_version, seq, val FROM crsql_changes "
            "WHERE val IN (10, 20, 30) ORDER BY db_version, seq"
        ).fetchall()

        print("  Sync consumer view (db_version, seq, val):")
        for ch in changes:
            print(f"    {ch}")

        # Check each db_version has unique seq values
        by_version = {}
        for db_v, seq, val in changes:
            by_version.setdefault(db_v, []).append(seq)

        for db_v, seqs in by_version.items():
            if len(seqs) != len(set(seqs)):
                print(f"  PROBLEM: db_version {db_v} has duplicate seqs: {seqs}")
            else:
                print(f"  db_version {db_v}: seqs are unique ({seqs})")

        close(c)


# ===========================================================================
# #431 — Missing seq of 0 on INSERT OR REPLACE
# https://github.com/vlcn-io/cr-sqlite/issues/431
# ===========================================================================


class TestIssue431MissingSeqZero:
    """
    INSERT OR REPLACE produces clock entries starting at seq=1 instead of
    seq=0 because the replace triggers a delete+insert internally.
    """

    def test_insert_or_replace_seq_starts_at_zero(self):
        """
        Exact reproduction from the issue.
        """
        c = connect(":memory:")
        c.execute(
            "CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, text1 TEXT, text2 TEXT, num1 INTEGER, num2 INTEGER) STRICT;"
        )
        c.execute("SELECT crsql_as_crr('foo')")
        c.commit()

        c.execute("INSERT INTO foo VALUES (1, 'foo', 'bar', 2, 3)")
        c.commit()

        # Check seq values after initial insert
        clock_v1 = c.execute(
            "SELECT col_name, seq FROM foo__crsql_clock WHERE key = 1 ORDER BY seq"
        ).fetchall()
        seqs_v1 = [r[1] for r in clock_v1]
        print(f"  After INSERT — seqs: {seqs_v1}")
        has_zero_v1 = 0 in seqs_v1

        # Now INSERT OR REPLACE
        c.execute("INSERT OR REPLACE INTO foo VALUES (1, 'foo2', 'bar2', 4, 5)")
        c.commit()

        clock_v2 = c.execute(
            "SELECT col_name, seq FROM foo__crsql_clock WHERE key = 1 ORDER BY seq"
        ).fetchall()
        seqs_v2 = [r[1] for r in clock_v2]
        print(f"  After REPLACE — seqs: {seqs_v2}")
        has_zero_v2 = 0 in seqs_v2

        db_version = c.execute("SELECT crsql_db_version()").fetchone()[0]

        # Check crsql_changes for this db_version
        changes = c.execute(
            "SELECT cid, seq FROM crsql_changes WHERE db_version = ? ORDER BY seq",
            (db_version,)
        ).fetchall()
        change_seqs = [r[1] for r in changes]
        print(f"  Changes at db_version {db_version} — seqs: {change_seqs}")

        if 0 not in change_seqs:
            print("  CONFIRMED: seq=0 is missing after INSERT OR REPLACE")
        else:
            print("  seq=0 is present — issue may be fixed")

        close(c)

    def test_insert_or_replace_changes_still_syncable(self):
        """
        Even if seq is off-by-one, the changes should still sync correctly
        to another node.
        """
        sender = connect(":memory:")
        sender.execute(
            "CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, name TEXT, val INTEGER) STRICT;"
        )
        sender.execute("SELECT crsql_as_crr('foo')")
        sender.commit()

        sender.execute("INSERT INTO foo VALUES (1, 'original', 100)")
        sender.commit()
        sender.execute("INSERT OR REPLACE INTO foo VALUES (1, 'replaced', 200)")
        sender.commit()

        receiver = connect(":memory:")
        receiver.execute(
            "CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, name TEXT, val INTEGER) STRICT;"
        )
        receiver.execute("SELECT crsql_as_crr('foo')")
        receiver.commit()

        changes = get_all_changes(sender)
        apply_changes(receiver, changes)

        sender_rows = get_table_contents(sender, "foo")
        receiver_rows = get_table_contents(receiver, "foo")
        assert sender_rows == receiver_rows, (
            f"Sync after REPLACE failed:\n"
            f"  sender: {sender_rows}\n"
            f"  receiver: {receiver_rows}"
        )

        close(sender)
        close(receiver)


# ===========================================================================
# #442 — Tables with >63 columns fail
# https://github.com/vlcn-io/cr-sqlite/issues/442
# ===========================================================================


class TestIssue442ColumnLimit:
    """
    crsql_as_crr fails with 'SQL logic error' when a table has more than
    63 non-PK columns. Likely a 64-bit bitmask overflow.
    """

    def test_63_columns_works(self):
        """63 columns (the reported limit) should work."""
        c = connect(":memory:")
        # 1 PK + 62 non-PK = 63 total columns
        cols = ", ".join(f"c{i} INTEGER" for i in range(62))
        c.execute(f"CREATE TABLE big (id INTEGER PRIMARY KEY NOT NULL, {cols});")

        try:
            c.execute("SELECT crsql_as_crr('big')")
            c.commit()
            print("  63 columns: SUCCEEDED")
        except Exception as e:
            pytest.fail(f"63 columns should work but failed: {e}")

        close(c)

    def test_64_columns_fails(self):
        """64 columns (one over the limit) reportedly fails."""
        c = connect(":memory:")
        # 1 PK + 63 non-PK = 64 total columns
        cols = ", ".join(f"c{i} INTEGER" for i in range(63))
        c.execute(f"CREATE TABLE big (id INTEGER PRIMARY KEY NOT NULL, {cols});")

        try:
            c.execute("SELECT crsql_as_crr('big')")
            c.commit()
            print("  64 columns: SUCCEEDED (issue may be fixed)")
        except Exception as e:
            print(f"  64 columns: FAILED — {e}")

        close(c)

    def test_100_columns(self):
        """Well over the limit — 100 columns."""
        c = connect(":memory:")
        cols = ", ".join(f"c{i} INTEGER" for i in range(99))
        c.execute(f"CREATE TABLE big (id INTEGER PRIMARY KEY NOT NULL, {cols});")

        try:
            c.execute("SELECT crsql_as_crr('big')")
            c.commit()

            # If it works, verify basic operations
            vals = ", ".join(["1"] + [str(i) for i in range(99)])
            c.execute(f"INSERT INTO big VALUES ({vals})")
            c.commit()

            changes = get_all_changes(c)
            print(f"  100 columns: SUCCEEDED — {len(changes)} changes generated")
        except Exception as e:
            print(f"  100 columns: FAILED — {e}")

        close(c)


# ===========================================================================
# #222 — NOT NULL columns without defaults fail on merge
# https://github.com/vlcn-io/cr-sqlite/issues/222
# ===========================================================================


class TestIssue222NotNullMerge:
    """
    CRDT merges happen column-by-column. If a column is NOT NULL without a
    DEFAULT, intermediate states during merge violate the constraint.
    """

    def test_not_null_with_default_works(self):
        """NOT NULL columns with defaults should merge fine."""
        sender = connect(":memory:")
        sender.execute(
            "CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, name TEXT NOT NULL DEFAULT '') STRICT;"
        )
        sender.execute("SELECT crsql_as_crr('foo')")
        sender.commit()

        sender.execute("INSERT INTO foo VALUES (1, 'hello')")
        sender.commit()

        receiver = connect(":memory:")
        receiver.execute(
            "CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, name TEXT NOT NULL DEFAULT '') STRICT;"
        )
        receiver.execute("SELECT crsql_as_crr('foo')")
        receiver.commit()

        changes = get_all_changes(sender)
        apply_changes(receiver, changes)

        assert get_table_contents(receiver, "foo") == [(1, 'hello')]

        close(sender)
        close(receiver)

    def test_not_null_without_default_rejected_at_crr_creation(self):
        """
        NOT NULL without DEFAULT — cr-sqlite rejects this at crsql_as_crr
        time, before you even get to a merge. This is the guard against
        the column-by-column merge issue.
        CONFIRMED: the error message is clear and actionable.
        """
        c = connect(":memory:")
        c.execute(
            "CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, name TEXT NOT NULL, score INTEGER NOT NULL) STRICT;"
        )

        with pytest.raises(Exception, match="NOT NULL column without a DEFAULT VALUE"):
            c.execute("SELECT crsql_as_crr('foo')")

        close(c)

    def test_nullable_columns_always_work(self):
        """Nullable columns should always merge fine (baseline)."""
        sender = connect(":memory:")
        sender.execute(
            "CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, name TEXT, score INTEGER) STRICT;"
        )
        sender.execute("SELECT crsql_as_crr('foo')")
        sender.commit()

        sender.execute("INSERT INTO foo VALUES (1, 'test', 42)")
        sender.commit()

        receiver = connect(":memory:")
        receiver.execute(
            "CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, name TEXT, score INTEGER) STRICT;"
        )
        receiver.execute("SELECT crsql_as_crr('foo')")
        receiver.commit()

        changes = get_all_changes(sender)
        apply_changes(receiver, changes)

        assert get_table_contents(receiver, "foo") == [(1, 'test', 42)]

        close(sender)
        close(receiver)
