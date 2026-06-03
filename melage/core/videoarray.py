import cv2
import numpy as np
import threading
import time
import os
from collections import OrderedDict
import re

class VideoLabelProxy:
    def __init__(self, parent_video_proxy, label_path=None, buffer_size=60):
        self.video = parent_video_proxy
        self.label_path = label_path

        # Geometry: (H, W, Frames) - Explicitly 2D per frame
        self.shape = self.video.shape[:3]
        self.ndim = 3
        self.dtype = np.uint8

        # A set to store integer indices of frames that contain segmentation
        self.segmented_indices = set()

        # 1. Memory Layer (User Edits) - Fast O(1)
        self.sparse_data = {}

        # 2. Disk Layer (Buffered) Setup
        self._cap = None
        self.label_files = []
        self.use_disk = False
        self.is_video_source = False
        self._is_valid = True
        if self.label_path:
            if os.path.isfile(self.label_path):
                # CASE A: Video File
                cap_temp = cv2.VideoCapture(self.label_path, cv2.CAP_FFMPEG)
                if cap_temp.isOpened():
                    self.use_disk = True
                    self.is_video_source = True
                    cap_temp.release()
                    self._cap = cv2.VideoCapture(self.label_path, cv2.CAP_FFMPEG)
                else:
                    print(f"Warning: Could not open label video {self.label_path}")

            elif os.path.isdir(self.label_path):
                # CASE B: Directory of Images
                valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
                seg_phto_files = sorted([
                    os.path.join(self.label_path, f)
                    for f in os.listdir(self.label_path)
                    if f.lower().endswith(valid_exts)
                ])
                is_source_video = self.video.is_video
                if is_source_video:
                    sorted_seg_files = seg_phto_files
                    verified = False
                    if len(sorted_seg_files)==self.video.shape[-2]:
                        verified = True
                else:
                    photo_path_files = self.video.image_files
                    sorted_seg_files, verified = self._verify_reorder_seg_files(seg_phto_files, photo_path_files, )
                if verified:
                    self.use_disk = True
                    self.is_video_source = False
                    self.label_files = sorted_seg_files
                    # We know exactly which frames have files
                    for idx, f_path in enumerate(self.label_files):
                        if f_path is not None:
                            self.segmented_indices.add(idx)
                    # Warning if lengths mismatch
                    if len(sorted_seg_files) != self.shape[2]:
                        print(f"Warning: Label count ({len(sorted_seg_files)}) != Video frames ({self.shape[2]})")
                        self._is_valid = False
                else:
                    self._is_valid = False
                    print(f"Warning: No images found in {self.label_path}")

        # --- BUFFER SETUP ---
        if self.use_disk:
            self.buffer_size = buffer_size
            self.buffer = OrderedDict()
            self.lock = threading.Lock()
            self.stop_event = threading.Event()
            self.worker_idx = 0
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()


    def _verify_reorder_seg_files(self, seg_phto_files, photo_path_files):
        seg_map = {
            os.path.splitext(os.path.basename(f))[0]: f
            for f in seg_phto_files
        }

        sorted_seg_files = []
        verified = True
        # 2. Iterate through photos to build the new sorted list
        for photo_path in photo_path_files:
            # Extract the name of the photo (e.g., "path/to/image_01.jpg" -> "image_01")
            photo_name = os.path.splitext(os.path.basename(photo_path))[0]

            # Define the expected segmentation key
            expected_seg_key = f"{photo_name}_seg"

            # 3. Check if the key exists in our map
            sorted_seg_files.append(seg_map.get(expected_seg_key, None))

        # Result: sorted_seg_files is now 1:1 aligned with photo_path_files

        return sorted_seg_files, verified,

    def save(self, output_path, fps=30):
        """
        Saves segmentation. Detects format based on output_path extension.
        - Ends in .avi/.mp4 -> Saves single video file.
        - Ends in .png/.jpg -> Saves sequence of images (created in a folder).
        - No extension -> Creates folder and saves as .png sequence.
        """

        # Determine Mode
        ext = os.path.splitext(output_path)[1].lower()
        is_video_output = ext in ['.avi', '.mp4', '.mov', '.mkv']

        if is_video_output:
            self._save_as_video(output_path, fps)
        else:
            self._save_as_sequence(output_path)

    def _save_as_video(self, output_path, fps):
        print(f"Saving video to {output_path} at {fps} FPS...")
        fourcc = cv2.VideoWriter_fourcc(*'png ')  # Lossless for AVI
        if output_path.endswith('.mp4'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer = cv2.VideoWriter(
            output_path, fourcc, fps,
            (self.shape[1], self.shape[0]), isColor=False
        )

        if not writer.isOpened():
            print(f"Error: Could not open writer for {output_path}")
            return

        for i in range(self.shape[2]):
            mask_2d = self.get_frame(i)
            writer.write(mask_2d)
            if i % 50 == 0: print(f"Saving frame {i}/{self.shape[2]}", end='\r')

        writer.release()
        print(f"\nSaved video successfully.")

    def _update_basename(self, base_name):
        name_part, extension = base_name.rsplit('.', 1)
        match = re.search(r'_(\d+)$', name_part)
        new_filename = base_name
        if match:
            # Get the number found
            number_str = match.group(1)

            # Pad the number to 6 digits
            # The int() conversion ensures it handles "01" correctly before re-padding
            padded_number = f"{int(number_str):06d}"

            # Replace the old number with the new padded number in the name
            # We slice name_part up to the start of the match, then add the new suffix
            new_name_part = name_part[:match.start(1)] + padded_number

            # Reassemble the filename
            new_filename = f"{new_name_part}.{extension}"
        return new_filename
    def get_label_frame(self, index):
        if self.video.is_video:
            formatted_index = f"{index:06d}"
            return formatted_index
        else:
            image_names = self.video.image_files
            if 0 <= index < len(image_names):

                base_name = os.path.basename(image_names[index])
                base_name = self._update_basename(base_name)
                image_name_no_ext = base_name.split('.')[0]
                return image_name_no_ext
            else:
                formatted_index = f"{index:06d}"
                return formatted_index
    def _save_as_sequence(self, output_path):
        """Saves as frame_00000.png, etc."""
        # Check if output_path is a specific file (e.g. data/labels.png) or a dir
        ext = os.path.splitext(output_path)[1]

        if ext:
            # User gave "folder/labels.png", treat "folder" as target, "labels" as prefix
            folder = os.path.dirname(output_path)
            prefix = os.path.basename(output_path).replace(ext, "")
            img_ext = ext
        else:
            # User gave "folder/", treat as target dir
            folder = output_path
            prefix = "frame"
            img_ext = ".png"

        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        print(f"Saving sequence to {folder}...")

        for i in range(self.shape[2]):
            mask_2d = self.get_frame(i)

            # Skip saving completely empty frames to save disk space?
            # Usually better to save all for consistency, but optional.
            filename = f"{prefix}_{i:05d}{img_ext}"
            fullpath = os.path.join(folder, filename)

            cv2.imwrite(fullpath, mask_2d)

            if i % 50 == 0: print(f"Saving {i}/{self.shape[2]}", end='\r')

        print(f"\nSaved sequence successfully.")

    def _worker(self):
        """Background thread filling the buffer."""
        while not self.stop_event.is_set():
            # 1. Check Buffer Limit
            with self.lock:
                should_sleep = (len(self.buffer) >= self.buffer_size) or \
                               (self.worker_idx >= self.shape[2])

            if should_sleep:
                time.sleep(0.01)
                continue

            # 2. Read Frame
            with self.lock:
                # Double-check index in case it changed while sleeping
                if self.worker_idx >= self.shape[2]:
                    continue

                if self.is_video_source:
                    # Video Read
                    ret, frame = self._cap.read()
                    if ret and frame.ndim == 3: frame = frame[..., 0]
                else:
                    # Sequence Read
                    if self.worker_idx < len(self.label_files):
                        # Force Grayscale to ensure 2D (H, W)
                        selected_file = self.label_files[self.worker_idx]
                        if selected_file is None:
                            ret, frame = False, None
                        else:
                            frame = cv2.imread(selected_file, cv2.IMREAD_GRAYSCALE)
                            ret = frame is not None
                    else:
                        ret, frame = False, None

                if ret:
                    self.buffer[self.worker_idx] = frame
                    while len(self.buffer) > self.buffer_size:
                        self.buffer.popitem(last=False)
                    self.worker_idx += 1
                else:
                    # Read failure or EOF
                    pass

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

        # LAYER 1: MEMORY
        if index in self.sparse_data:
            return self.sparse_data[index]

        # LAYER 2: DISK
        if self.use_disk:
            # Hit
            with self.lock:
                frame = self.buffer.get(index)
                if frame is not None:
                    return frame

            # Miss -> Hard Seek
            with self.lock:
                self.buffer.clear()
                self.worker_idx = index

                ret = False
                frame = None

                if self.is_video_source:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ret, frame = self._cap.read()
                    if ret and frame.ndim == 3: frame = frame[..., 0]
                else:
                    if 0 <= index < len(self.label_files):
                        selected_file = self.label_files[index]
                        if selected_file is None:
                            ret, frame = False, None
                        else:
                            frame = cv2.imread(self.label_files[index], cv2.IMREAD_GRAYSCALE)
                            ret = frame is not None

                if ret:
                    self.buffer[index] = frame
                    self.worker_idx += 1
                    return frame

        # LAYER 3: EMPTY
        return np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)

    def close(self):
        if self.use_disk:
            self.stop_event.set()
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
            if self.is_video_source and self._cap:
                self._cap.release()

    # --- Standard Boilerplate ---
    def get_data_dtype(self):
        return np.dtype(self.dtype)

    def get_fdata(self):
        return self

    def _expand_slice(self, slicer):
        if not isinstance(slicer, tuple): slicer = (slicer,)
        if Ellipsis in slicer:
            idx = slicer.index(Ellipsis)
            missing = 3 - (len(slicer) - 1)
            slicer = slicer[:idx] + (slice(None),) * missing + slicer[idx + 1:]
        elif len(slicer) < 3:
            slicer = slicer + (slice(None),) * (3 - len(slicer))
        return slicer

    def get_metadata(self):
        """
        Returns a comprehensive dictionary of metadata.
        """
        # 1. Basic Counts
        # Count frames that have segmentation (from disk files + memory edits)
        valid_seg_count = len(self.segmented_indices)

        # 2. Source Type
        if self.is_video_source:
            src_type = "video_file"
        elif self.use_disk:
            src_type = "image_sequence"
        else:
            src_type = "empty/new"

        # 3. File System Info (Size & Dates)
        file_size_mb = 0.0
        file_date = "Unknown"

        try:
            if self.label_path and os.path.exists(self.label_path):
                stats = os.stat(self.label_path)
                # Size in MB
                file_size_mb = stats.st_size / (1024 * 1024)
                # Modification time (YYYY-MM-DD HH:MM:SS)
                import datetime
                dt = datetime.datetime.fromtimestamp(stats.st_mtime)
                file_date = dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            pass

        # 4. Video Specifics (FPS, Duration, Codec)
        fps = 0.0
        duration_sec = 0.0
        codec = "N/A"

        if self.video.is_video and self.video.path_source:
            # We need to momentarily open the video to query properties
            # (Or if you kept self.video.fps stored somewhere, use that)
            temp_cap = cv2.VideoCapture(self.video.path_source, cv2.CAP_FFMPEG)
            if temp_cap.isOpened():
                fps = temp_cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    frame_count = temp_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration_sec = frame_count / fps

                # Extract Codec (FourCC)
                # This returns a float, we need to convert it to string characters
                fourcc_int = int(temp_cap.get(cv2.CAP_PROP_FOURCC))
                codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

                temp_cap.release()

        # 5. Build the Dictionary
        meta = {
            "source_type": src_type,
            "width": self.shape[1],
            "height": self.shape[0],
            "num_frames": self.shape[2],
            "segmented_count": valid_seg_count,
            "progress_percent": round((valid_seg_count / self.shape[2]) * 100, 1) if self.shape[2] > 0 else 0,
            "label_path": self.label_path if self.label_path else "None",
            "image_path": self.video.path_source if self.video.path_source else "None",
            # New Fields
            "file_size_mb": round(file_size_mb, 2),
            "last_modified": file_date,
            "fps": round(fps, 2),
            "duration_sec": round(duration_sec, 2),
            "codec": codec
        }

        return meta
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
                # 1. User is ADDING segmentation
                if data.ndim == 3: data = data[:, :, 0]
                self.sparse_data[frame_idx] = data.astype(np.uint8)

                # Update Index Set
                self.segmented_indices.add(frame_idx)
            else:
                # 2. User is CLEARING segmentation (Eraser)
                if frame_idx in self.sparse_data:
                    del self.sparse_data[frame_idx]

                # If clearing a frame that exists on disk, we explicitly store zeros
                if self.use_disk:
                    self.sparse_data[frame_idx] = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)

                # Update Index Set: Remove from set because it is now empty
                self.segmented_indices.discard(frame_idx)

class VideoArrayProxy:
    def __init__(self, path_source, buffer_size=60, fps=30):
        """
        path_source: Can be a video file path (str), a directory of images (str),
                     or a list of image file paths (list).
        fps: Used only if loading an image sequence (default 30).
        """
        self.path_source = path_source
        self.image_files = []
        self.is_video = False


        # --- 1. Detect Input Type & Initialize Metadata ---
        if isinstance(path_source, (list, tuple)):
            # CASE A: List of file paths
            self.image_files = sorted(path_source)
            self._init_sequence(fps)
        elif os.path.isdir(path_source):
            # CASE B: Directory of images
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
            files = [os.path.join(path_source, f) for f in os.listdir(path_source)
                     if f.lower().endswith(valid_exts)]
            self.image_files = sorted(files)
            if not self.image_files:
                raise IOError(f"No images found in folder: {path_source}")
            self._init_sequence(fps)
        else:
            # CASE C: Video File
            self.is_video = True
            self._init_video(path_source)

        # Common Metadata
        self.shape = (self.height, self.width, self.frames, 3)
        self.ndim = 4
        self.dtype = np.uint8

        # --- 2. Buffering Setup ---
        self.buffer_size = buffer_size
        self.buffer = OrderedDict()
        self.sparse_data = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker_idx = 0
        self.current_index = 0
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def get_label_frame(self, index):
        if self.is_video:
            formatted_index = f"{index:06d}"
            return formatted_index
        else:
            if 0 <= index < len(self.image_files):
                base_name = os.path.basename(self.image_files[index])
                image_name_no_ext = base_name.split('.')[0]
                return image_name_no_ext
            else:
                formatted_index = f"{index:06d}"
                return formatted_index

    def _init_video(self, file_path):
        """Helper to setup video capture metadata"""
        self._cap = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {file_path}")
        self.frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        # Note: We keep self._cap open for the worker

    def _init_sequence(self, fps):
        """Helper to setup image sequence metadata"""
        self.frames = len(self.image_files)
        self.fps = fps

        # Read first image to set the "Authoritative" dimensions
        first_img = cv2.imread(self.image_files[0])
        if first_img is None:
            raise IOError(f"Failed to read first image: {self.image_files[0]}")

        self.height, self.width = first_img.shape[:2]
        self.fixed_size = (self.width, self.height)  # Store (W, H) for cv2.resize
        self._cap = None

    def _validate_frame(self, frame):
        """
        Ensures frame matches the established (Height, Width).
        Resizes if dimensions differ (e.g., accidental thumbnails mixed in).
        """
        if frame is None: return None

        h, w = frame.shape[:2]
        if (w, h) != self.fixed_size:
            # Enforce consistency
            return cv2.resize(frame, self.fixed_size, interpolation=cv2.INTER_LINEAR)

        return frame

    def _read_next_frame_internal(self):
        if self.is_video:
            return self._cap.read()
        else:
            if self.worker_idx >= self.frames:
                return False, None
            frame = cv2.imread(self.image_files[self.worker_idx])
            ret = frame is not None

            if ret:
                frame = self._validate_frame(frame)  # <--- Add this check

            return ret, frame

    def _worker(self):
        """Background thread filling the OrderedDict."""
        while not self.stop_event.is_set():
            # 1. Check Buffer Size
            with self.lock:
                # If buffer is full, we stop "producing"
                # If we are at the end of the video, we also stop "producing"
                should_sleep = (len(self.buffer) >= self.buffer_size) or \
                               (self.worker_idx >= self.frames)

            # 2. SLEEP if needed
            if should_sleep:
                time.sleep(0.01)
                continue

            # 3. Read Frame
            with self.lock:
                # Double-check index inside lock
                if self.worker_idx >= self.frames:
                    continue

                # Unified read logic
                ret, frame = self._read_next_frame_internal()

                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.buffer[self.worker_idx] = frame

                    # FIFO Maintenance
                    while len(self.buffer) > self.buffer_size:
                        self.buffer.popitem(last=False)

                    self.worker_idx += 1
                else:
                    pass  # EOF or Read Error

            if not ret:
                time.sleep(0.01)

    def get_frame(self, index):
        if index < 0: index += self.frames
        self.current_index = index
        if index in self.sparse_data:
            return self.sparse_data[index]
        # 1. INSTANT DICT LOOKUP
        with self.lock:
            frame = self.buffer.get(index)
            if frame is not None:
                return frame

                # 2. MISS -> Hard Seek & Reset
        with self.lock:
            self.buffer.clear()
            self.worker_idx = index  # Reset worker target

            if self.is_video:
                # Video: Must physically seek the capture object
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ret, frame = self._cap.read()
            else:
                if 0 <= index < self.frames:
                    frame = cv2.imread(self.image_files[index])
                    ret = frame is not None
                    if ret:
                        frame = self._validate_frame(frame)  # <--- Add this check
                else:
                    ret, frame = False, None

            if not ret:
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.buffer[index] = frame
            self.worker_idx += 1  # Advance worker to next frame
            return frame

    def __setitem__(self, slicer, data):
        """
        Allows modifying specific regions of a frame.
        Example: proxy[y_slice, x_slice, frame_index, c_slice] = new_data
        """
        full_slice = self._expand_slice(slicer)
        frame_idx = full_slice[2]

        if isinstance(frame_idx, int):
            # 1. Fetch the current frame (from sparse edits or disk/buffer)
            # We use .copy() so we don't accidentally mutate the read-only buffer
            current_frame = self.get_frame(frame_idx).copy()

            # 2. Extract the spatial and channel slices (Y, X, C)
            # full_slice is (y_slice, x_slice, t_slice, c_slice)
            frame_slicer = (full_slice[0], full_slice[1], full_slice[3])

            # 3. Apply the modification
            current_frame[frame_slicer] = data

            # 4. Save the modified frame into the fast memory dictionary
            self.sparse_data[frame_idx] = current_frame
    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.is_video and self._cap:
            self._cap.release()

    # ... [Same helper methods as before: get_data_dtype, get_fdata, _expand_slice, __getitem__] ...
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
        return np.zeros((1, 1, 1, 3), dtype=np.uint8)