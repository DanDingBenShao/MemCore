#!/usr/bin/env python3
"""Test MemoryCore disk mode with larger dataset"""
import os
import sys
import tempfile
import shutil
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_core import MemoryCore

def generate_test_data(n=500):
    """Generate n test memories"""
    templates = [
        ("weather {}", "Today in {}, weather is {}, temperature {}C"),
        ("meeting {}", "Meeting with {} about {} at {}pm"),
        ("project {}", "Project {} status: {}, next step is {}"),
        ("bug {}", "Bug in {}: {}, fixed by {}"),
        ("task {}", "Task {} priority {}: {}"),
    ]
    
    cities = ["Beijing", "Shanghai", "Shenzhen", "Hangzhou", "Chengdu"]
    weather = ["sunny", "cloudy", "rainy", "windy"]
    people = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    topics = ["product", "design", "tech", "marketing"]
    status = ["ongoing", "blocked", "done", "review"]
    
    data = []
    for i in range(n):
        t = random.choice(templates)
        if "weather" in t[0]:
            q = t[0].format(i)
            c = t[1].format(random.choice(cities), random.choice(weather), random.randint(15, 35))
            fp = f"weather {random.choice(cities)}"
        elif "meeting" in t[0]:
            q = t[0].format(i)
            c = t[1].format(random.choice(people), random.choice(topics), random.randint(1, 6))
            fp = f"meeting {random.choice(people)}"
        else:
            q = t[0].format(i)
            c = t[1].format(i, random.choice(status), f"action{i}")
            fp = f"task{i}"
        data.append((q, c, fp))
    return data

def test_performance():
    test_dir = tempfile.mkdtemp(prefix="memcore_perf_test_")
    print(f"Test dir: {test_dir}")
    
    try:
        # Test with 500 memories
        n_memories = 500
        
        db_file = os.path.join(test_dir, "test.db")
        disk_file = os.path.join(test_dir, "disk_vectors.db")
        
        print(f"\n[1] Creating MemoryCore with {n_memories} memories...")
        mc = MemoryCore(
            db_path=db_file,
            max_memories=10000,
            use_disk_store=True,
            disk_store_path=disk_file
        )
        
        # Generate and add memories
        data = generate_test_data(n_memories)
        
        start = time.time()
        for q, c, fp in data:
            mc.add_memory(q, c, fp)
        add_time = time.time() - start
        print(f"[1] Added {n_memories} memories in {add_time:.2f}s ({add_time/n_memories*1000:.1f}ms per memory)")
        
        print(f"[1] Total memories: {len(mc.memories)}")
        print(f"[1] DiskStore vectors: {len(mc.disk_store)}")
        print(f"[1] self.X empty: {len(mc.X) == 0}")
        
        # Test recall performance
        queries = [
            "weather Beijing",
            "meeting Alice",
            "project status",
            "bug fix",
            "task priority",
        ]
        
        print(f"\n[2] Testing recall performance...")
        for q in queries:
            start = time.time()
            results = mc.recall(q)
            recall_time = (time.time() - start) * 1000
            print(f"    Query '{q}': {len(results)} results in {recall_time:.1f}ms")
        
        # Test multiple recalls
        print(f"\n[3] Testing 100 recalls...")
        start = time.time()
        for _ in range(100):
            mc.recall(random.choice(queries))
        total_time = time.time() - start
        avg_time = total_time / 100 * 1000
        print(f"[3] 100 recalls in {total_time:.2f}s, avg {avg_time:.1f}ms per recall")
        
        # Get stats
        stats = mc.disk_store.get_stats()
        print(f"\n[4] DiskStore stats:")
        print(f"    Total vectors: {stats['total_vectors']}")
        print(f"    Memory MB: {stats.get('memory_mb', 'N/A')}")
        
        print("\n[PASS] Performance test completed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
        print("[Cleanup] Removed test dir")

if __name__ == "__main__":
    test_performance()
