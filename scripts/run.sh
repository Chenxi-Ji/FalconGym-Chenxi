#!/bin/bash

# Ensure the script exits on any error
set -e

# Check if case_name is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <case_name>"
    exit 1
fi

case_name=$1
cd /home/chenxij2/FalconGym-Chenxi

# Run generate_3D_gsplat.py
# echo "Running generate_3D_gsplat.py..."
# python3 scripts/generate_3D_gsplat.py --config configs/${case_name}/3d.yaml
# echo "This may take a while depending on the size of the point cloud. You are recommended to open (ns-viewer) the gsplat scene in preview."

# Run generate_traj.py
echo "Running generate_traj.py..."
MPLBACKEND=Agg python3 scripts/generate_traj.py --config configs/${case_name}/traj.yaml
echo ""

# Update render.yaml for "traj"
echo "Setting filename to 'traj' in configs/${case_name}/render.yaml..."
sed -i 's/filename:.*/filename: traj/' configs/${case_name}/render.yaml

# Run traj_render.py for "traj"
echo "Running traj_render.py for traj..."
python3 scripts/traj_render.py --config configs/${case_name}/render.yaml
echo ""

# Update render.yaml for "samples"
echo "Setting filename to 'samples' in configs/${case_name}/render.yaml..."
sed -i 's/filename:.*/filename: samples/' configs/${case_name}/render.yaml

# Run traj_render.py for "samples"
echo "Running traj_render.py for samples..."
python3 scripts/traj_render.py --config configs/${case_name}/render.yaml
echo ""

echo "All scripts executed successfully!"
