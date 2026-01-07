import cv2
import numpy as np
import threading
import time
from collections import OrderedDict


class VideoLabelProxy:
    def __init__(self, parent_video_proxy, label_file_path=None, buffer_size=60):
        self.video = parent_video_proxy
        self.label_file_path = label_file_path

        # Geometry: (H, W, Frames) - Explicitly 2D per frame
        self.shape = self.video.shape[:3]
        self.ndim = 3
        self.dtype = np.uint8

        # 1. Memory Layer (User Edits) - Fast O(1)
        self.sparse_data = {}

        # 2. Disk Layer (Buffered)
        self._cap = None
        self.use_disk = False

        if self.label_file_path:
            cap_temp = cv2.VideoCapture(self.label_file_path)
            if cap_temp.isOpened():
                self.use_disk = True
                cap_temp.release()
                self._cap = cv2.VideoCapture(self.label_file_path)
            else:
                print(f"Warning: Could not open label video {self.label_file_path}")


        # --- OPTIMIZED BUFFER SETUP ---
        if self.use_disk:
            self.buffer_size = buffer_size

            # USE ORDERED DICT FOR O(1) ACCESS
            self.buffer = OrderedDict()

            self.lock = threading.Lock()
            self.stop_event = threading.Event()
            self.worker_idx = 0

            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def save(self, output_path, fps):
        """
        Saves segmentation as a Lossless Grayscale AVI using original video's FPS.
        """
        if not output_path.endswith('.avi'):
            output_path = output_path.rsplit('.', 1)[0] + '.avi'


        print(f"Saving to {output_path} at {fps} FPS...")

        fourcc = cv2.VideoWriter_fourcc(*'png ')  # Lossless

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,  # <--- Use the exact source FPS
            (self.shape[1], self.shape[0]),
            isColor=False
        )

        if not writer.isOpened():
            print(f"Error: Could not open writer for {output_path}")
            return

        for i in range(self.shape[2]):
            mask_2d = self.get_frame(i)
            if mask_2d.sum()>0:
                print('There are more than 0 values in the mask')
            writer.write(mask_2d)
            if i % 50 == 0: print(f"Saving frame {i}/{self.shape[2]}", end='\r')

        writer.release()
        print(f"\nSaved successfully.")

    def _worker(self):
        """Background thread filling the OrderedDict. Optimized to prevent CPU spikes."""
        while not self.stop_event.is_set():
            # 1. Check Condition (Atomic/Locked)
            with self.lock:
                # Sleep if buffer is full OR we reached the end of the video
                should_sleep = (len(self.buffer) >= self.buffer_size) or \
                               (self.worker_idx >= self.shape[2])

            # 2. SLEEP if Idle (Crucial Fix)
            # This prevents the "while True: continue" busy loop at the end of the video
            if should_sleep:
                time.sleep(0.01)
                continue

            # 3. Read & Process
            with self.lock:
                # Double-check index in case it changed while sleeping
                if self.worker_idx >= self.shape[2]:
                    continue

                ret, frame = self._cap.read()

                if ret:
                    # OPTIMIZATION: Convert to 2D immediately to save RAM
                    if frame.ndim == 3:
                        # Taking just one channel is faster than cvtColor for labels
                        # Assuming grayscale/binary info is replicated across channels
                        frame = frame[..., 0]

                        # Store in Dict (Instant Access)
                    self.buffer[self.worker_idx] = frame

                    # FIFO Removal: Remove oldest if full
                    while len(self.buffer) > self.buffer_size:
                        self.buffer.popitem(last=False)

                    self.worker_idx += 1
                else:
                    # Read failed (EOF or corrupt frame)
                    pass

            # 4. Handle Read Failure Sleep
            # If ret was False, we sleep outside the lock to avoid hammering the disk
            if not ret:
                time.sleep(0.01)


    def get_frame(self, index):
        """
        Priority:
        1. Sparse Data (Memory) - Instant O(1)
        2. Buffer (Pre-fetched Disk) - Instant O(1)
        3. Hard Seek (Disk Miss) - ~30ms
        4. Empty - Instant
        """
        if index < 0: index += self.shape[2]

        # --- LAYER 1: MEMORY (User Edits) ---
        if index in self.sparse_data:
            return self.sparse_data[index]

        # --- LAYER 2: DISK (Existing File) ---
        if self.use_disk:
            # A. Check Buffer (O(1) Lookup)
            with self.lock:
                frame = self.buffer.get(index)
                if frame is not None:
                    return frame  # Success: ~0.06ms

            # B. Buffer Miss -> Hard Seek
            with self.lock:
                self.buffer.clear()
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                self.worker_idx = index

                ret, frame = self._cap.read()
                if ret:
                    if frame.ndim == 3:
                        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        #_, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
                        #frame = frame//255
                        frame = frame[...,0]
                    self.buffer[index] = frame
                    self.worker_idx += 1
                    return frame

        # --- LAYER 3: EMPTY ---
        return np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)

    def close(self):
        if self.use_disk:
            self.stop_event.set()
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
            self._cap.release()

    def get_data_dtype(self):
        return np.dtype(self.dtype)

    def get_fdata(self):
        return self

    # --- Standard Slicing ---
    def _expand_slice(self, slicer):
        if not isinstance(slicer, tuple): slicer = (slicer,)
        if Ellipsis in slicer:
            idx = slicer.index(Ellipsis)
            missing = 3 - (len(slicer) - 1)
            slicer = slicer[:idx] + (slice(None),) * missing + slicer[idx + 1:]
        elif len(slicer) < 3:
            slicer = slicer + (slice(None),) * (3 - len(slicer))
        return slicer

    def __getitem__(self, slicer):
        full_slice = self._expand_slice(slicer)
        frame_idx = full_slice[2]

        if isinstance(frame_idx, int):
            full_frame = self.get_frame(frame_idx)
            return full_frame[full_slice[0], full_slice[1]]

        return np.zeros((1, 1, 1), dtype=np.uint8)

    def __setitem__(self, slicer, data):
        full_slice = self._expand_slice(slicer)
        frame_idx = full_slice[2]

        if isinstance(frame_idx, int):
            if np.any(data):
                if data.ndim == 3: data = data[:, :, 0]
                self.sparse_data[frame_idx] = data.astype(np.uint8)
            else:
                if frame_idx in self.sparse_data:
                    del self.sparse_data[frame_idx]
                if self.use_disk:
                    self.sparse_data[frame_idx] = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)





class VideoArrayProxy:
    def __init__(self, file_path, buffer_size=60):
        self.file_path = file_path

        # 1. Standard Metadata
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened(): raise IOError(f"Cannot open: {file_path}")
        self.frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        self.shape = (self.height, self.width, self.frames, 3)
        self.ndim = 4
        self.dtype = np.uint8

        # 2. Buffering Setup
        self._cap = cv2.VideoCapture(file_path)
        self.buffer_size = buffer_size

        # OPTIMIZATION: Use OrderedDict instead of Deque
        # OrderedDict allows O(1) lookup AND keeps order for FIFO removal
        self.buffer = OrderedDict()

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker_idx = 0

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        """Background thread filling the OrderedDict."""
        while not self.stop_event.is_set():
            # 1. Check Buffer Size
            with self.lock:
                # If buffer is full, we stop "producing"
                # If we are at the end of the video, we also stop "producing"
                should_sleep = (len(self.buffer) >= self.buffer_size) or \
                               (self.worker_idx >= self.frames)

            # 2. SLEEP if there is nothing to do (Crucial Fix!)
            # Prevents 100% CPU usage loop at the end of video
            if should_sleep:
                time.sleep(0.01)
                continue

            # 3. Read Frame
            with self.lock:
                # Double-check index inside lock (in case a seek happened during sleep)
                if self.worker_idx >= self.frames:
                    continue

                ret, frame = self._cap.read()

                if ret:
                    # Success: Process frame
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.buffer[self.worker_idx] = frame

                    # FIFO Maintenance: Pop oldest
                    while len(self.buffer) > self.buffer_size:
                        self.buffer.popitem(last=False)

                    self.worker_idx += 1
                else:
                    # Failure (End of File): Release Lock & Sleep
                    # This prevents the 'else: pass' tight loop
                    pass

            # If read failed (EOF), sleep a bit to avoid hammering the disk/CPU
            if not ret:
                time.sleep(0.01)


    def get_frame(self, index):
        """
        Optimized Getter: O(1) Lookup
        """
        if index < 0: index += self.frames

        # 1. INSTANT DICT LOOKUP (No Looping)
        with self.lock:
            # .get() is O(1) - Extremely fast
            frame = self.buffer.get(index)

            if frame is not None:
                return frame  # Success: ~0.06ms

        # 2. MISS -> Hard Seek & Reset
        with self.lock:
            self.buffer.clear()  # Reset cache
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.worker_idx = index

            ret, frame = self._cap.read()
            if not ret: return np.zeros((self.height, self.width, 3), dtype=np.uint8)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.buffer[index] = frame
            self.worker_idx += 1
            return frame

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self._cap.release()

    def get_data_dtype(self):
        return np.dtype(self.dtype)

    def get_fdata(self):
        return self

    def _expand_slice(self, slicer):
        if not isinstance(slicer, tuple): slicer = (slicer,)
        if Ellipsis in slicer:
            idx = slicer.index(Ellipsis)
            missing = 4 - (len(slicer) - 1)
            slicer = slicer[:idx] + (slice(None),) * missing + slicer[idx + 1:]
        elif len(slicer) < 4:
            slicer = slicer + (slice(None),) * (4 - len(slicer))
        return slicer

    def __getitem__(self, slicer):
        full_slice = self._expand_slice(slicer)
        frame_idx = full_slice[2]

        if isinstance(frame_idx, int):
            return self.get_frame(frame_idx)[full_slice[0], full_slice[1]]

        if isinstance(frame_idx, slice):
            # ... (slice handling logic) ...
            return np.zeros((1, 1, 1, 3), dtype=np.uint8)