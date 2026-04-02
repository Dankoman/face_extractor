import sqlite3
from pathlib import Path
from collections import Counter

def get_stats(db_path):
    conn = sqlite3.connect(db_path)
    # 1. Total counts
    total_processed = conn.execute("SELECT COUNT(*) FROM processed").fetchone()[0]
    total_aliases = conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]

    # 2. OK vs Fail
    ok_fail = dict(conn.execute("SELECT ok, COUNT(*) FROM processed GROUP BY ok").fetchall())
    ok_count = ok_fail.get(1, 0)
    fail_count = ok_fail.get(0, 0)

    # 3. Top failure reasons
    reasons = conn.execute("SELECT reason, COUNT(*) as c FROM processed WHERE ok=0 GROUP BY reason ORDER BY c DESC LIMIT 10").fetchall()

    # 4. Extracting person names and stats
    # We iterate over unique paths or a sample if it's too large.
    # Given 68MB, it's likely around 500k rows. Iterating might take a few seconds.
    # We'll use SQLite's group by if possible, but extracting person from path is easier in Python.
    # Alternatively, we can use SQLite's substr/instr but it varies per path structure.
    
    person_counts = Counter()
    person_fails = Counter()
    
    # Let's get all paths and process them in Python for a more complete picture.
    # If this is too slow, we can sample.
    cursor = conn.execute("SELECT path, ok FROM processed")
    unique_persons = set()
    
    for path, ok in cursor:
        try:
            # Assumes path format: /.../PersonName/filename.ext
            person = Path(path).parent.name
            unique_persons.add(person)
            person_counts[person] += 1
            if not ok:
                person_fails[person] += 1
        except Exception:
            continue

    # 5. Top Persons (most images)
    top_persons_images = person_counts.most_common(10)
    
    # 6. Top Persons (most failures)
    top_persons_fails = person_fails.most_common(10)

    # 7. Alias details
    # People with most aliases
    most_aliases = conn.execute("SELECT primary_name, COUNT(*) as c FROM aliases GROUP BY primary_name ORDER BY c DESC LIMIT 10").fetchall()

    conn.close()

    return {
        "total_processed": total_processed,
        "total_aliases": total_aliases,
        "ok_count": ok_count,
        "fail_count": fail_count,
        "reasons": reasons,
        "unique_persons": len(unique_persons),
        "top_persons_images": top_persons_images,
        "top_persons_fails": top_persons_fails,
        "most_aliases": most_aliases
    }

if __name__ == "__main__":
    db = "arcface_work-ppic/processed.db"
    stats = get_stats(db)
    
    print("--- Database Statistics ---")
    print(f"Total Processed Images: {stats['total_processed']:,}")
    print(f"Total Unique Persons:    {stats['unique_persons']:,}")
    print(f"Total Aliases:          {stats['total_aliases']:,}")
    print(f"\nProcessing Status:")
    print(f"  OK:   {stats['ok_count']:,} ({(stats['ok_count']/stats['total_processed'])*100:.1f}%)")
    print(f"  FAIL: {stats['fail_count']:,} ({(stats['fail_count']/stats['total_processed'])*100:.1f}%)")
    
    print("\nTop 5 Failure Reasons:")
    for reason, count in stats['reasons'][:5]:
        print(f"  - {reason}: {count:,}")

    print("\nTop 5 Persons (Most Images):")
    for person, count in stats['top_persons_images'][:5]:
        print(f"  - {person}: {count:,} images")

    print("\nTop 5 Persons (Most Failures):")
    for person, count in stats['top_persons_fails'][:5]:
        print(f"  - {person}: {count:,} failures")

    print("\nTop 5 Primary Names by Alias Count:")
    for name, count in stats['most_aliases'][:5]:
        print(f"  - {name}: {count:,} aliases")
