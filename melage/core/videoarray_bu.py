import cv2
import numpy as np

from functools import lru_cache
class VideoLabelProxy:
    def __init__(self, parent_video_proxy, label_file_path=None):
        self.video = parent_video_proxy
        self.label_file_path = label_file_path

        # 1. Enforce Shape: (Height, Width, Frames)
        # We explicitly slice [:3] to ignore the 4th dim (Channels) of the video
        # Video shape is (H, W, T, 3) -> Label shape becomes (H, W, T)
        self.shape = self.video.shape[:3]
        self.ndim = 3
        self.dtype = np.uint8

        self.sparse_data = {}
        self._cap = None

        if self.label_file_path:
            self._cap = cv2.VideoCapture(self.label_file_path)

    @lru_cache(maxsize=1280)
    def get_frame(self, index):
        """
        Returns a guaranteed 2D array (H, W).
        """
        # Handle negative indexing
        if index < 0: index += self.shape[2]

        # 1. Memory Layer
        if index in self.sparse_data:
            return self.sparse_data[index]

        # 2. Disk Layer
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = self._cap.read()
            if ret:
                # CRITICAL: Strip RGB if present
                if frame.ndim == 3:
                    # Assume mask is grayscale or red-channel.
                    # Using [:,:,0] extracts just one channel (2D)
                    return frame[:, :, 0]
                return frame  # Already 2D

        # 3. Empty (2D Black Frame)
        return np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)

    def __getitem__(self, slicer):
        # ... (Same expand_slice logic as before) ...
        # Helper to handle '...'
        if not isinstance(slicer, tuple): slicer = (slicer,)
        if Ellipsis in slicer:
            idx = slicer.index(Ellipsis)
            missing = 3 - (len(slicer) - 1)
            slicer = slicer[:idx] + (slice(None),) * missing + slicer[idx + 1:]
        elif len(slicer) < 3:
            slicer = slicer + (slice(None),) * (3 - len(slicer))

        frame_idx = slicer[2]

        if isinstance(frame_idx, int):
            full_frame = self.get_frame(frame_idx)
            return full_frame[slicer[0], slicer[1]]

        return np.zeros((1, 1, 1), dtype=np.uint8)

    def __setitem__(self, slicer, data):
        """
        Enforce storing ONLY 2D data.
        """
        # ... (Same expand_slice logic) ...
        if not isinstance(slicer, tuple): slicer = (slicer,)
        if Ellipsis in slicer:
            idx = slicer.index(Ellipsis)
            missing = 3 - (len(slicer) - 1)
            slicer = slicer[:idx] + (slice(None),) * missing + slicer[idx + 1:]
        elif len(slicer) < 3:
            slicer = slicer + (slice(None),) * (3 - len(slicer))

        frame_idx = slicer[2]

        if isinstance(frame_idx, int):
            if np.any(data):
                # SAFETY CHECK: If input is RGB (H, W, 3), flatten it
                if data.ndim == 3:
                    # Take the first channel, or max projection if multi-color
                    data = data[:, :, 0]

                # Ensure it is exactly 2D before saving
                if data.ndim == 2:
                    self.sparse_data[frame_idx] = data.astype(np.uint8)
            else:
                if frame_idx in self.sparse_data:
                    del self.sparse_data[frame_idx]
                if self._cap is not None:
                    # Store 2D Zeros
                    self.sparse_data[frame_idx] = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)

    def get_fdata(self):
        return self
    def get_data_dtype(self):
        """
        Mimics the Nibabel/NumPy API.
        Returns the numpy dtype object (usually np.uint8 for video).
        """
        return np.dtype(self.dtype)

class VideoArrayProxy:
    """
    A lightweight proxy that looks like a 4D NumPy array (H, W, Time, RGB)
    but reads video frames from disk on-demand.
    """

    def __init__(self, file_path):
        self.file_path = file_path

        # Open once to get metadata, then close
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {file_path}")

        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # The internal CV2 capture object (kept open for slicing)
        self._cap = cv2.VideoCapture(file_path)

        # Medical Orientation: (Height, Width, Depth/Time, Channels)
        self.shape = (self.height, self.width, self.frame_count, 3)
        self.ndim = 4
        self.dtype = np.uint8

    def _expand_slice(self, slicer):
        """
        Correctly expands Ellipsis (...) into slice(None) to match 4D shape.
        """
        if not isinstance(slicer, tuple):
            slicer = (slicer,)

        # 1. Check for Ellipsis
        if Ellipsis in slicer:
            # Only support one Ellipsis (standard python limit)
            idx = slicer.index(Ellipsis)

            # How many dims do we need to fill?
            # Total dims (4) - (Items in slicer - 1 for the ellipsis)
            missing = 4 - (len(slicer) - 1)

            # Create the expanded tuple
            # Part before ... + (:, :, ...) + Part after ...
            slicer = slicer[:idx] + (slice(None),) * missing + slicer[idx + 1:]

        # 2. Pad with (:) if tuple is too short (e.g., data[0])
        elif len(slicer) < 4:
            slicer = slicer + (slice(None),) * (4 - len(slicer))

        return slicer

    def __getitem__(self, slicer):
        # 1. Expand the slice correctly (Fixes the slow bug)
        full_slice = self._expand_slice(slicer)

        # 2. Extract Indices
        # Shape is (Height, Width, Time/Depth, RGB)
        # We need the 3rd element (Time)
        frame_idx = full_slice[2]

        # Optimization: Single Frame (Fast)
        if isinstance(frame_idx, int):
            return self._read_frame(frame_idx, full_slice)

        # Optimization: Slice (Loop)
        elif isinstance(frame_idx, slice):
            start = frame_idx.start or 0
            stop = frame_idx.stop or self.frame_count
            step = frame_idx.step or 1

            # Safety: Don't try to read 80,000 frames if someone asks for [:]
            if (stop - start) // step > 500:
                print(f"Warning: Attempting to read {stop - start} frames. This will be slow.")

            stack = []
            for i in range(start, stop, step):
                # Read frame i, preserving X/Y/C slicing
                # Create a sub-slice: (x, y, i, c)
                sub_slice = (full_slice[0], full_slice[1], i, full_slice[3])
                stack.append(self._read_frame(i, sub_slice))

            if not stack:
                return np.zeros((0, 0, 0), dtype=np.uint8)

            return np.stack(stack, axis=2)

        return None

    @lru_cache(maxsize=1280)
    def get_frame(self, index):
        """Directly reads a single frame. O(1) speed."""
        return self._read_frame(index, (slice(None), slice(None), index, slice(None)))
    def _read_frame(self, index, slicer):
        """Helper to read one specific frame"""
        # Handle negative indexing
        if index < 0: index += self.frame_count

        # Seek to frame (This is the slow part, but unavoidable for random access)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()

        if not ret:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Convert BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Transpose (H, W, 3) is good, but we need to match the slice request
        # The proxy shape is (H, W, Z, C), but the frame is (H, W, C).
        # We apply the X (0), Y (1), and Channel (3) slices.
        x_sl, y_sl, _, c_sl = slicer
        return frame[x_sl, y_sl, c_sl]

    def get_data_dtype(self):
        """
        Mimics the Nibabel/NumPy API.
        Returns the numpy dtype object (usually np.uint8 for video).
        """
        return np.dtype(self.dtype)
    def get_fdata(self):
        """
        DANGER: If something calls this, it loads the WHOLE video.
        We return self to try and trick generic readers, or implement chunking.
        """
        return self

    def __array__(self):
        """Called if someone forces np.array(proxy)"""
        # Warn user or load chunk?
        print("Warning: Loading entire video into RAM...")
        # Load all frames (Legacy behavior)
        return self[:]