import os
from pathlib import Path
from identity_resolver import IdentityResolver
from external_resolver import ExternalIdentityResolver

def test_logic():
    print("--- 1. Testing Transitive Aliases ---")
    # We'll use temporary files for the test
    with open("test_merge.txt", "w") as f:
        f.write("Alice|Alias1\n")
        f.write("Alias1|Alias2\n")
    
    with open("test_exclusions.txt", "w") as f:
        f.write("Bob|NotBob\n")
    
    resolver = IdentityResolver("test_merge.txt", "test_exclusions.txt")
    
    print(f"Alice group: {resolver.get_group('Alice')}")
    assert "Alias2" in resolver.get_group("Alice")
    assert resolver.resolve("Alias2") == "Alice" # Fallback to local primary (first in list)
    
    print("\n--- 2. Testing External Truth & Priority ---")
    # Feed some 'truths'
    resolver.add_external_truth("Alice", "Queen Alice", "ThePornDB")
    print(f"Resolve Alice (TPDB): {resolver.resolve('Alice')}")
    assert resolver.resolve("Alice") == "Queen Alice"
    
    resolver.add_external_truth("Alias1", "Grand Queen Alice", "StashDB")
    print(f"Resolve Alice (StashDB wins): {resolver.resolve('Alice')}")
    assert resolver.resolve("Alice") == "Grand Queen Alice"
    
    print("\n--- 3. Testing Conflict Detection ---")
    # Add a conflict: Bob and NotBob are linked as same, but excluded
    with open("test_merge_conflict.txt", "w") as f:
        f.write("Bob|NotBob\n")
    
    resolver_conflict = IdentityResolver("test_merge_conflict.txt", "test_exclusions.txt")
    conflicts = resolver_conflict.check_conflicts()
    print(f"Conflicts found: {conflicts}")
    assert len(conflicts) > 0

    print("\n✅ All logic tests passed!")
    
    # Cleanup
    os.remove("test_merge.txt")
    os.remove("test_exclusions.txt")
    os.remove("test_merge_conflict.txt")

if __name__ == "__main__":
    test_logic()
