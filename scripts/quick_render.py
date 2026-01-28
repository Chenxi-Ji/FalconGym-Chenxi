import os

# # Set the CUDA architecture list to match your GPU's compute capability (e.g., 8.6 for Ampere GPUs)
# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

import torch

# Enable TensorFloat32 for better performance on Ampere GPUs and newer
torch.set_float32_matmul_precision('high')

import PIL.Image
import numpy as np
from gsplat.rendering import rasterization, _rasterization
import matplotlib.pyplot as plt
import time
from nerfstudio.utils import colormaps
from scipy.spatial.transform import Rotation
import json
import cv2
import pickle
import sys


# NOTE: check https://docs.gsplat.studio/main/apis/rasterization.html
# NOTE: also check nerfstudio source code splatfacto.py 
#  to see customized usage of rasterization(), get_viewmat()

def get_viewmat(optimized_camera_to_world, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
   """
   function that converts c2w to gsplat world2camera matrix, using compile for some speed
   """
   R = optimized_camera_to_world[:, :3, :3].to(device) # 3 x 3
   T = optimized_camera_to_world[:, :3, 3:4].to(device)  # 3 x 1
   # flip the z and y axes to align with gsplat conventions
   R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
   # analytic matrix inverse to get world2camera matrix
   R_inv = R.transpose(1, 2)
   T_inv = -torch.bmm(R_inv, T)
   viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
   viewmat[:, 3, 3] = 1.0  # homogenous
   viewmat[:, :3, :3] = R_inv
   viewmat[:, :3, 3:4] = T_inv
   return viewmat

def render_single_frame(pose, means, quats, opacities, scales, colors,
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),  
                        width = 640, height = 480, fx = 546.84164912, fy = 547.57957461, 
                        cx = 349.18316327, cy = 215.54486004,
                        # width=1024, height=768, fx=506.7404229124384, fy=506.26636309344354,
                        # cx=511.7028853190299, cy=376.7415082018125,
                        gsplat_path = "outputs/multi_gates/splatfacto/2025-05-27_140811",
                        visualize = False, scale=1, debug=False):
   # render
   s = time.time()

   # Scale can be used to play with different resolution
   width, height = int(width*scale), int(height*scale)
   fx, fy, cx, cy = fx*scale, fy*scale, cx*scale, cy*scale
   Ks = torch.tensor([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], device=device)[None, :, :]

   # define cameras
   viewmats = np.eye(4)

   # Step 1: Construct 4x4 matrix in real-world coordinate system
   # pose = np.array([2.5, -15, -0.5, 1.57, 0, 0])
   px, py, pz, yaw, pitch, roll = pose
   rot = Rotation.from_euler("ZYX", (yaw, pitch, roll))
   R_matrix = rot.as_matrix()
   viewmats[:3, :3] = R_matrix  # Set the top-left 3x3 to the rotation matrix
   viewmats[:3, 3] = [px,py,pz]  # Set the translation components
   # viewmats = torch.from_numpy(viewmats).to(device)

   # Step 2: convert from real-world coordinates to transmform.json coordinate
   # Convert camera pose to what's stated in transforms_orig.json
   tmp = Rotation.from_euler('zyx',[-np.pi/2,np.pi/2,0]).as_matrix()
   mat = viewmats[:3,:3]@tmp 
   viewmats[:3,:3] = mat 

   # # Convert camera pose to Colmap frame in transforms.json
   viewmats[0:3,1:3] *= -1
   viewmats = viewmats[np.array([0,2,1,3]),:]
   viewmats[2,:] *= -1 

   with open(f"{gsplat_path}/dataparser_transforms.json", 'r') as f:
      dp_trans_info = json.load(f)
   transform = np.array(dp_trans_info['transform'])
   scale_factor = dp_trans_info['scale']
   viewmats = transform@viewmats
   viewmats[:3,3] *= scale_factor
   viewmats = viewmats[:3,:]
   if viewmats.ndim == 2:
      viewmats = np.expand_dims(viewmats, axis=0)
   viewmats = torch.FloatTensor( viewmats ).to(device)


   viewmats = get_viewmat(viewmats)

   render, alpha, meta = rasterization(
      means, 
      quats, 
      scales=torch.exp(scales),
      opacities=torch.sigmoid(opacities).squeeze(-1), 
      colors=colors, 
      viewmats=viewmats, 
      Ks=Ks, 
      width=width, 
      height=height,
      packed = False,
      near_plane=0.01,
      far_plane=1e10,
      render_mode="RGB+ED",
      sh_degree=3,
      sparse_grad=False,
      absgrad=True,
      rasterize_mode="classic",
   )

   # NOTE: to be consistent with NeRFStudio splatfacto rendering
   # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
   background = torch.tensor([0.1490, 0.1647, 0.2157]).to(device)
   alpha = alpha[:, ...]  
   rgb = render[:, ..., :3] + (1 - alpha) * background
   rgb = torch.clamp(rgb, 0.0, 1.0)

   # print (meta.keys())

   img = rgb.squeeze(0)

   img =(colormaps.apply_colormap(image=img, colormap_options=colormaps.ColormapOptions(colormap='gray', normalize=True, colormap_min=0, colormap_max=255) )).cpu().numpy()
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   img = (img * 255).astype(np.uint8)
   img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

   if debug:
      print(f"Time taken: {round(time.time()-s,4)}s")
   s = time.time()

   cv_img = img[:, :, ::-1]
   if visualize:
      cv2.imwrite(f'figures/{visualize}', cv_img)
   return cv_img

if __name__ == "__main__":

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(device)

   script_dir = os.path.dirname(os.path.realpath(__file__))
   output_folder = os.path.join(script_dir, '.')
   gsplat_path = "outputs/uturn/splatfacto/2025-05-09_151825"
   checkpoint = f"{gsplat_path}/nerfstudio_models/step-000039999.ckpt"

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

   print(f"Shift Time: {round(time.time()-s, 4)}s")

   for _ in range(1):
      # px, py, pz, yaw, pitch, roll
      pose = [0, -7, -0.2, 1.57, 0, 0]
      render_single_frame(pose, means, quats, opacities, scales, colors, device,
               gsplat_path=gsplat_path, visualize='test.png', debug=True
      )

   # # ############ Generate Training Image for UNet Gate Detection ##############
   # import random

   # for idx in range(3000):
   #    x = random.random() * 4 - 3 #[-3, 1]
   #    y = random.random() * 6 - 5.5 #[-0.5, -5.5] 
   #    z = random.random() * 2 - 1 #[-1, 1]
   #    yaw = random.random() * 0.6 + (np.pi-0.3) # [2.84, 3.44]
   #    pitch = random.random() * 0.4 - 0.2 #[-0.2, 0.2]
   #    roll = 0

   #    pose = [x, y, z, yaw, pitch, roll]
   #    img = render_single_frame(pose, means, quats, opacities, scales, colors, device,
   #             gsplat_path=gsplat_path, visualize=True, debug=True
   #    )
   #    cv2.imwrite(f"./gate_detection/UNet-Pytorch-Customdataset/gate_imgs/{x}_{y}_{z}_{yaw}_{pitch}_{roll}.png", img)
   # # ############ Generate Training Image for UNet Gate Detection ##############


   # # # ############ Quick render the txt ##############
   # with open('traj.txt', 'r') as f:
   #    contents = f.readlines()

   # import os
   # os.system('rm trajectory_images/*.png')
   # for idx, log in enumerate(contents):
   #    if idx % 5:
   #       continue
   #    x, y, z, yaw, pitch, roll = [float(i) for i in log.split(" ")]
   #    # Conversion from dynamics coordinates to gsplat coordinates
   #    pitch, roll = roll, pitch
   #    pitch *= -1
   #    # roll *= -1
   #    pose = [x, y, z, yaw, pitch, roll]
   #    img = render_single_frame(pose, means, quats, opacities, scales, colors, device,
   #             gsplat_path=gsplat_path, visualize=True, debug=True
   #    )
   #    print(idx)
   #    cv2.imwrite(f"./trajectory_images/{idx:04}.png", img)
   # # # ############ Quick render txt ##############
