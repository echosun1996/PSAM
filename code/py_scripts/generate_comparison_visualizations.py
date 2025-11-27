#!/usr/bin/env python3
"""
Script to generate visualization images for a specific HAM10000 image
from multiple PSAM result directories and save them to a new directory.

Usage:
    python generate_comparison_visualizations.py \
        --image-filename ISIC_0024312.jpg \
        --dataset val \
        --output-dir /path/to/output \
        --color Oranges
"""

import os
import argparse
import subprocess
import re
import tempfile
import shutil
from pathlib import Path


def generate_visualization_for_directory(
    source_dir,
    image_filename,
    dataset,
    color,
    parent_dir,
    output_base_dir
):
    """
    Generate visualization for a specific image from a source directory.
    
    Args:
        source_dir: Source directory path (e.g., PSAM_cp_1_P_B)
        image_filename: Name of the image file (e.g., ISIC_0024312.jpg)
        dataset: Dataset split (val or test)
        color: Colormap name for visualization
        parent_dir: Parent directory of the project
        output_base_dir: Base output directory
    """
    # Extract directory name (e.g., PSAM_cp_1_P_B)
    dir_name = os.path.basename(source_dir.rstrip('/'))
    
    # Paths
    mask_input_path = os.path.join(source_dir, "HAM10000", dataset, "mask")
    image_path = os.path.join(parent_dir, "data", "HAM10000", "input", dataset, "HAM10000_img")
    points_csv_path = os.path.join(parent_dir, "data", "HAM10000", "input", f"HAM10000_{dataset}_prompts_42.csv")
    
    # Output directory for this specific source directory
    output_dir = os.path.join(output_base_dir, dir_name)
    
    # Check if mask directory exists
    if not os.path.exists(mask_input_path):
        print(f"⚠️  Mask directory not found: {mask_input_path}")
        return False
    
    # Check if the required mask files exist for this image
    base_name = os.path.splitext(image_filename)[0]
    
    # Check if token visualization files exist (for uncertain_vis.py)
    has_token_vis_files = all([
        os.path.exists(os.path.join(mask_input_path, f"{base_name}_uncertain.png")),
        os.path.exists(os.path.join(mask_input_path, f"{base_name}_ps.png")),
        os.path.exists(os.path.join(mask_input_path, f"{base_name}_ns.png"))
    ])
    
    # Check if at least the main mask file exists
    main_mask_file = os.path.join(mask_input_path, f"{base_name}.jpg")
    if not os.path.exists(main_mask_file):
        print(f"⚠️  Main mask file not found: {main_mask_file}")
        return False
    
    # If token visualization files don't exist, we'll add pentagrams to existing visualization
    if not has_token_vis_files:
        print(f"ℹ️  Token visualization files not found in {mask_input_path}")
        print(f"   Will add pentagram markers to existing visualization...")
        
        # Check if there's a vis directory with the image
        vis_dir = os.path.join(os.path.dirname(mask_input_path), "vis")
        vis_file = os.path.join(vis_dir, image_filename)
        
        if os.path.exists(vis_file):
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, image_filename)
            
            # Get paths for adding pentagrams
            original_image_path = os.path.join(
                parent_dir, "data", "HAM10000", "input", dataset, "HAM10000_img", image_filename
            )
            points_csv_path = os.path.join(
                parent_dir, "data", "HAM10000", "input", f"HAM10000_{dataset}_prompts_42.csv"
            )
            
            # Add pentagrams using the new script
            add_pentagrams_script = os.path.join(
                parent_dir, "code", "py_scripts", "add_pentagrams_to_vis.py"
            )
            
            cmd = [
                "python", add_pentagrams_script,
                "--vis-image", vis_file,
                "--original-image", original_image_path,
                "--csv-path", points_csv_path,
                "--output-path", output_file,
                "--image-filename", image_filename
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and os.path.exists(output_file):
                print(f"✅ Added pentagrams to visualization: {output_file}")
                return True
            else:
                print(f"⚠️  Failed to add pentagrams: {result.stderr}")
                # Fallback: just copy the file
                shutil.copy2(vis_file, output_file)
                print(f"   Copied original visualization instead")
                return True
        else:
            print(f"⚠️  Visualization file not found: {vis_file}")
            print(f"   This directory may not have been evaluated with visualization enabled.")
            return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a temporary directory for this specific image
    temp_output_dir = os.path.join(output_dir, "uncertain_vis")
    os.makedirs(temp_output_dir, exist_ok=True)
    
    # Call uncertain_vis.py - we need to modify it to only process this specific image
    # Since uncertain_vis.py with revised=True uses a hardcoded list, we'll create
    # a modified version that processes only this image
    script_path = os.path.join(parent_dir, "code", "py_scripts", "figs", "uncertain_vis.py")
    
    print(f"🔄 Generating visualization for {dir_name}...")
    print(f"   Input: {mask_input_path}")
    print(f"   Output: {temp_output_dir}")
    print(f"   Image: {image_filename}")
    
    try:
        # Read the script and modify it temporarily
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # Create a modified version that only processes this image
        # Replace the file_names list in the revised section
        lines = script_content.split('\n')
        modified_lines = []
        in_revised_section = False
        in_file_names_list = False
        
        for i, line in enumerate(lines):
            if 'if args.revised:' in line:
                in_revised_section = True
                modified_lines.append(line)
            elif in_revised_section and 'file_names = [' in line:
                in_file_names_list = True
                modified_lines.append('    file_names = [')
                modified_lines.append(f'        "{base_name}",')
            elif in_file_names_list and line.strip() == ']':
                modified_lines.append('    ]')
                in_file_names_list = False
                in_revised_section = False
            elif in_file_names_list:
                # Skip lines inside the file_names list
                continue
            else:
                modified_lines.append(line)
        
        modified_content = '\n'.join(modified_lines)
        
        # Write to a temporary file
        temp_script = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
        temp_script.write(modified_content)
        temp_script.close()
        
        # Execute the modified script
        cmd = [
            "python", temp_script.name,
            "--predict-input", mask_input_path,
            "--intergrated-output", temp_output_dir,
            "--image", image_path,
            "--points-csv", points_csv_path,
            "--color", color,
            "--revised", "True"
        ]
        
        result = subprocess.run(
            cmd,
            cwd=os.path.join(parent_dir, "code", "py_scripts", "figs"),
            capture_output=True,
            text=True
        )
        
        # Clean up temp file
        try:
            os.unlink(temp_script.name)
        except:
            pass
        
        if result.returncode != 0:
            print(f"❌ Error generating visualization for {dir_name}:")
            print(result.stderr)
            return False
        
        # Check if output file was created
        expected_output = os.path.join(
            temp_output_dir,
            f"{base_name}_{color}_intergration.jpg"
        )
        
        if os.path.exists(expected_output):
            print(f"✅ Successfully generated: {expected_output}")
            return True
        else:
            print(f"⚠️  Expected output file not found: {expected_output}")
            return False
            
    except Exception as e:
        print(f"❌ Exception while generating visualization for {dir_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization images for a specific HAM10000 image from multiple PSAM result directories"
    )
    parser.add_argument(
        "--image-filename",
        type=str,
        required=True,
        help="Name of the image file (e.g., ISIC_0024312.jpg)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split (val or test)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save all visualizations"
    )
    parser.add_argument(
        "--color",
        type=str,
        default="Oranges",
        help="Colormap name for visualization (default: Oranges)"
    )
    parser.add_argument(
        "--parent-dir",
        type=str,
        default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        help="Parent directory of the PSAM project"
    )
    parser.add_argument(
        "--source-dirs",
        type=str,
        nargs="+",
        default=[
            "PSAM_cp_1_P_B",
            "PSAM_cp_2_P_B",
            "PSAM_cp_3_P_B",
            "PSAM_cp_4_B",
            "PSAM_cp_4_P",
            "PSAM_cp_4_P_B",
            "PSAM_cp_5_P_B",
            "PSAM_cp_6_P_B",
            "PSAM_cp_7_P_B",
            "PSAM_cp_8_P_B",
            "PSAM_cp_9_P_B",
            "PSAM_cp_10_P_B",
        ],
        help="List of source directory names (relative to results/)"
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    results_base = os.path.join(args.parent_dir, "results")
    source_dirs_full = [
        os.path.join(results_base, dir_name)
        for dir_name in args.source_dirs
    ]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"Generating visualizations for image: {args.image_filename}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Color map: {args.color}")
    print(f"Source directories: {len(source_dirs_full)}")
    print("=" * 80)
    
    # Process each source directory
    success_count = 0
    failed_dirs = []
    
    for source_dir in source_dirs_full:
        dir_name = os.path.basename(source_dir)
        if generate_visualization_for_directory(
            source_dir,
            args.image_filename,
            args.dataset,
            args.color,
            args.parent_dir,
            args.output_dir
        ):
            success_count += 1
        else:
            failed_dirs.append(dir_name)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Successfully processed: {success_count}/{len(source_dirs_full)}")
    if failed_dirs:
        print(f"  Failed directories: {', '.join(failed_dirs)}")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

