# melage/core/headless.py
import sys
import nibabel as nib
from melage.core.io import load_image_core
from melage.utils.headless_utils import list_available_tools, get_plugin_runner


def save_result(result, output_path, reference_header=None):
    import numpy as np
    if isinstance(result, nib.Nifti1Image):
        nib.save(result, output_path)
    elif isinstance(result, np.ndarray):
        affine = reference_header.get_best_affine() if reference_header is not None else np.eye(4)
        nib.save(nib.Nifti1Image(result, affine), output_path)
    else:
        print(f"Warning: Unknown result type {type(result)}, skipping save.")


def run_headless_mode(args):
    """
    Main entry point for headless execution.
    """
    print(f"--- MELAGE Headless: {args.tool} ---")

    # 1. Validation
    if not args.input or not args.output:
        print("Error: Input and Output paths are required.")
        sys.exit(1)

    # 2. Load Data
    readIM, info, fmt = load_image_core(args.input)
    if not info[1]:
        print(f"Error loading file: {info[2]}")
        sys.exit(1)

    # 3. Resolve Plugin
    plugin_func = get_plugin_runner(args.tool)

    if not plugin_func:
        print(f"Error: Could not find runner for tool '{args.tool}'")
        sys.exit(1)

    # 4. Execute
    try:
        result = plugin_func(readIM.im, readIM.affine, args)

        # 5. Save
        save_result(result, args.output, readIM.im.header)

    except Exception as e:
        print(f"CRITICAL FAILURE during processing: {e}")
        sys.exit(1)