import torch
import numpy as np
from gsplat.rendering import rasterization, _rasterization
import matplotlib.pyplot as plt
import os
import time
from nerfstudio.utils import colormaps
from scipy.spatial.transform import Rotation
import json
import cv2
import time
import pickle

from traj_render import render_single_frame
from utils import world2ckpt_xyz, ckpt2world_xyz, world2ckpt_xyz_batch, ckpt2world_xyz_batch
from utils import batch_quat_ckpt_to_world_matrix, batch_rotm_world_to_quat_ckpt

from edit_gsplat_api import translate, delete, duplicate, rotate, add, scale_func

import argparse
import yaml


def construct_gate_track(
      gate_pkl_file='objects_IDs/base_gate_id_in_big_ellipse.pkl',
      gsplat_path="outputs/multi_gates/splatfacto/2025-05-27_140811", 
      load_checkpoint="step-000039999.ckpt",
      to_save_checkpoint=None,
      gate_poses=None):
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   # print(device)

   script_dir = os.path.dirname(os.path.realpath(__file__))
   output_folder = os.path.join(script_dir, '.')

   checkpoint = f"{gsplat_path}/nerfstudio_models/{load_checkpoint}"

   checkpoint_fn = os.path.join(output_folder, '.', checkpoint)
   res = torch.load(checkpoint_fn)
   means = res['pipeline']['_model.gauss_params.means']
   quats = res['pipeline']['_model.gauss_params.quats']
   opacities = res['pipeline']['_model.gauss_params.opacities']
   scales = res['pipeline']['_model.gauss_params.scales']

   # Note we need both base color (dc) and direction dependent (rest)
   dc = res['pipeline']['_model.gauss_params.features_dc']
   rest = res['pipeline']['_model.gauss_params.features_rest']
   colors = torch.cat((dc[:, None, :], rest), dim=1)

   s = time.time()
   with open(gate_pkl_file, 'rb') as f:
      mask = pickle.load(f)      # returns the same torch.Tensor object

   # Gates
   if gate_poses is not None:
      last_yaw = 0
      for i, gate_pose in enumerate(gate_poses):
         pose = gate_pose[:3]
         yaw = gate_pose[3]
         yaw_diff = yaw - last_yaw
         
         if i==0:
            means, _, _ = translate(
               means,  mask=mask, global_pos=pose, debug=True, gsplat_path=gsplat_path
            )
            means, quats = rotate(means, quats, yaw=yaw_diff, mask=mask, gsplat_path=gsplat_path)

         else:
            means, quats, opacities, scales, colors, new_mask = duplicate(
               means, quats, opacities, scales, colors, yaw=yaw_diff, 
               global_pos=pose, mask=mask, debug=True, gsplat_path=gsplat_path
            )

            zeros_tensor = torch.zeros_like(mask)
            mask = torch.cat((zeros_tensor, new_mask), dim=0)

         last_yaw = yaw
         print(f"Shift Time: {round(time.time()-s, 4)}s")
   

   for _ in range(1):
      pose = np.array([0, -7, -0.2, 1.57, 0, 0]) # Front View
      render_single_frame(pose, means, quats, opacities, scales, colors, device,
               gsplat_path=gsplat_path, visualize='test.png', debug=True
      )

   ### Save new gsplat to a ckpt
   if to_save_checkpoint is not None:
      res['pipeline']['_model.gauss_params.means'] = means
      res['pipeline']['_model.gauss_params.quats'] = quats
      res['pipeline']['_model.gauss_params.opacities'] = opacities
      res['pipeline']['_model.gauss_params.scales'] = scales

      dc = colors[:, 0, :]
      rest = colors[:, 1:, :]
      res['pipeline']['_model.gauss_params.features_dc'] = dc
      res['pipeline']['_model.gauss_params.features_rest'] = rest

      new_checkpoint = f"{gsplat_path}/nerfstudio_models/{to_save_checkpoint}"
      new_checkpoint_fn = os.path.join(output_folder, '.', new_checkpoint)
      torch.save(res, new_checkpoint_fn)
      print("Saved new checkpoint:", new_checkpoint_fn)


if __name__ == "__main__":
   ### Default command: python3 scripts/generate_3D_gsplat.py --config configs/uturn/3d.yaml
   parser = argparse.ArgumentParser(description="Load configuration.")
   parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
   
   args = parser.parse_args()

   # Load parameters from YAML file
   with open(args.config, 'r') as file:
      config = yaml.safe_load(file)

   # Extract parameters for construct_gate_track
   gate_pkl_file = config["gate_pkl_file"]
   gsplat_path = config["gsplat_path"]
   load_checkpoint = config["load_checkpoint"]
   to_save_checkpoint = config["to_save_checkpoint"]
   gate_poses = config["gate_poses"]

   script_dir = os.path.dirname(os.path.realpath(__file__))
   gsplat_path = os.path.join(script_dir, '.', gsplat_path)

   # Call construct_gate_track with parameters from the YAML file
   construct_gate_track(
      gate_pkl_file=gate_pkl_file,
      gsplat_path=gsplat_path,
      load_checkpoint=load_checkpoint,
      to_save_checkpoint=to_save_checkpoint,
      gate_poses=gate_poses
   )
