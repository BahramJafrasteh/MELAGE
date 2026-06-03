import numpy as np

from test import post_process
import sys

def test_post_process():
    print("Testing post_process...")
    
    # Create a dummy segmentation with some noise/holes
    # Shape: (1, 1, 100, 100, 100) to match expected input format (Batch, Channel, D, H, W)
    # But post_process expects (1, 1, D, H, W) or similar, let's check usage.
    # In test.py: segmentation = prob_value.argmax(1).unsqueeze(1).float()
    # So input is (Batch, 1, D, H, W)
    
    shape = (1, 1, 50, 50, 50)
    seg = np.zeros(shape)
    
    # Create a main component for label 3
    seg[0, 0, 10:40, 10:40, 10:40] = 3
    
    # Create a small disconnected component for label 3 (should be removed)
    seg[0, 0, 5:8, 5:8, 5:8] = 3
    
    # Create a "hole" inside the main component (should be filled)
    seg[0, 0, 20:25, 20:25, 20:25] = 0
    
    # Create some other labels (background/other tissues)
    seg[0, 0, 40:45, 40:45, 40:45] = 1
    
    print(f"Original unique values: {np.unique(seg)}")
    print(f"Original label 3 count: {np.sum(seg == 3)}")
    
    # Run post_process
    # ind_whole=[3] means we process label 3
    processed_seg = post_process(seg, ind_whole=[3])
    
    print(f"Processed unique values: {np.unique(processed_seg)}")
    print(f"Processed label 3 count: {np.sum(processed_seg == 3)}")
    
    # Verification checks
    # 1. Small component should be removed (or filled with nearest neighbor if valid)
    # The small component at 5:8 is surrounded by 0 (background). 
    # If 0 is a valid label (not in ind_whole), it might be filled with 0.
    
    # 2. Hole should be filled
    hole_region = processed_seg[0, 0, 20:25, 20:25, 20:25]
    if np.all(hole_region == 3):
        print("PASS: Hole filled correctly.")
    else:
        print(f"FAIL: Hole not filled completely. Unique values in hole: {np.unique(hole_region)}")

    # 3. Small component check
    small_component = processed_seg[0, 0, 5:8, 5:8, 5:8]
    # It should NOT be 3 anymore if it was small enough.
    # Threshold is 5% of largest component.
    # Largest ~ 30*30*30 - hole ~ 27000 - 125 = 26875
    # Small ~ 3*3*3 = 27
    # 27 < 0.05 * 26875 (1343), so it should be removed.
    # It should be replaced by nearest neighbor (likely 0).
    
    if np.all(small_component != 3):
        print("PASS: Small component removed/replaced.")
    else:
        print("FAIL: Small component still present as label 3.")

if __name__ == "__main__":
    try:
        test_post_process()
        print("Test script finished successfully.")
    except Exception as e:
        print(f"Test script failed with error: {e}")
        import traceback
        traceback.print_exc()
