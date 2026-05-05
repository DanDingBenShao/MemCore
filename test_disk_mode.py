#!/usr/bin/env python3
"""Test MemoryCore disk mode"""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_core import MemoryCore

def test_disk_mode():
    test_dir = tempfile.mkdtemp(prefix="memcore_disk_test_")
    print(f"Test dir: {test_dir}")
    
    try:
        db_file = os.path.join(test_dir, "test.db")
        disk_file = os.path.join(test_dir, "disk_vectors.db")
        mc = MemoryCore(
            db_path=db_file,
            max_memories=100,
            use_disk_store=True,
            disk_store_path=disk_file
        )
        
        print(f"[1] Created, disk mode: {mc.use_disk_store}")
        print(f"[1] DiskStore initialized: {mc.disk_store is not None}")
        
        memories = [
            ("weather report", "Beijing sunny today, temp 25C", "Beijing weather sunny"),
            ("meeting note", "Product review meeting at 3pm", "meeting afternoon review"),
            ("tech note", "React uses virtual DOM for performance", "React DOM performance"),
        ]
        
        for query, content, fp in memories:
            mc.add_memory(query, content, fp)
            print(f"[2] Added: {query}")
        
        print(f"[2] Total memories: {len(mc.memories)}")
        print(f"[2] DiskStore vectors: {len(mc.disk_store)}")
        print(f"[2] self.X empty: {len(mc.X) == 0}")
        
        results = mc.recall("Beijing weather")
        print(f"[3] Query: 'Beijing weather', Results: {len(results)}")
        if results:
            print(f"[3] Top1: {results[0][:80]}...")
        
        results2 = mc.recall("React features")
        print(f"[4] Query: 'React features', Results: {len(results2)}")
        if results2:
            print(f"[4] Top1: {results2[0][:80]}...")
        
        print(f"[5] _disk_id_map entries: {len(mc._disk_id_map)}")
        
        print("\n[PASS] Disk mode basic test passed")
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
    test_disk_mode()
