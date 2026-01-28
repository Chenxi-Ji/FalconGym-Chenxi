
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

from quick_render import render_single_frame
from utils import world2ckpt_xyz, ckpt2world_xyz, world2ckpt_xyz_batch, ckpt2world_xyz_batch
from utils import batch_quat_ckpt_to_world_matrix, batch_rotm_world_to_quat_ckpt

def rotate(
   means: torch.Tensor,
   quats: torch.Tensor,
   yaw=0, pitch=0, roll=0,
   ux=0, lx=0,
   uy=0, ly=0,
   uz=0, lz=0,
   mask=None, debug=False, gsplat_path="outputs/multi_gates/splatfacto/2025-05-27_140811"
):
   """
   Rotate both means and quaternions of Gaussians inside a world space bbox.

   Args:
      means: (N,3) tensor of Gaussian centers in ckpt coords
      quats: (N,4) tensor of Gaussian orientations (w,x,y,z) in ckpt coords
      yaw,pitch,roll: rotation angles (radians) about world Z,Y,X axes
      lx,ux,ly,uy,lz,uz: floats defining the world space box bounds (use arg `mask` for segmented Gaussian masks)
      mask: (N,) boolean tensor, True for Gaussians to be edited (if None, default to world space bounding box with `lx,ux,ly,uy,lz,uz`)
      debug (optional): boolean for logging additional debugging information
      gsplat_path: path to nerfstudio model (eg. outputs/<model name>/splatfacto/<date>)

   Returns:
      new_means: (N,3) tensor of rotated centers in ckpt coords
      new_quats: (N,4) tensor of rotated quaternions in ckpt coords
   """
   # 1) ckpt → world
   world = ckpt2world_xyz_batch(means, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")  # (N,3)
   device = means.device

   # 2) build mask in world coordinates
   # NOTE:Either using bbox defined by user or 
   # (mask-id so that user does not need to track bbox)
   if mask is None:
      mask = (
         (world[:,0] >= lx) & (world[:,0] <= ux) &
         (world[:,1] >= ly) & (world[:,1] <= uy) &
         (world[:,2] >= lz) & (world[:,2] <= uz)
      )  # (N,)

   count = int(mask.sum().item())
   

   # 3) compute box center in world
   world_shifted = world.clone()
   if mask is None:
      center = torch.tensor(
         [(lx+ux)/2, (ly+uy)/2, (lz+uz)/2],
         device=device, dtype=world.dtype
      )  # (3,)
   else:
      center = world_shifted[mask, :].mean(dim=0)
   
   if debug:
      print("Gaussian within this bbox: ", count)
      print(f"Rotate around Center: {[round(p, 1) for p in center.tolist()]}")

   # 4) extract & center the masked points
   pts = world[mask] - center  # (M,3)

   # 5) world‐space rotation matrix (ZYX: yaw,pitch,roll)
   R = torch.tensor(
      Rotation.from_euler('ZYX', (yaw, pitch, roll)).as_matrix(),
      device=device, dtype=world.dtype
   )  # (3,3)

   # 6) rotate & re‐add pivot
   rotated_pts = (pts @ R.T) + center  # (M,3)

   # 7) scatter back into full world array
   world_rot = world.clone()
   world_rot[mask] = rotated_pts       # (N,3)

   new_means = world2ckpt_xyz_batch(world_rot, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")

   # 9) rotate quaternions for same subset
   new_quats = quats.clone()

   # Convert quat from ckpt to world space, apply transform & back to ckpt space
   R_world = batch_quat_ckpt_to_world_matrix(quats[mask], dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")
   R_world2 = R @ R_world
   new_quats[mask] = batch_rotm_world_to_quat_ckpt(R_world2, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")

   return new_means, new_quats


def delete(
    means: torch.Tensor,
    quats, opacities, scales, colors,
    ux=0, lx=0,
    uy=0, ly=0,
    uz=0, lz=0,
    mask=None, debug=False, gsplat_path="outputs/multi_gates/splatfacto/2025-05-27_140811"
):
   """
   Return the new tensor of Gaussians with the masked or bounding boxed regions of Gaussians removed 
   
   means: (N,3) tensor in ckpt coords
   quats: (N,4) tensor of Gaussian orientations (w,x,y,z) in ckpt coords
   opacities: (N,1) tensor of Gaussian opacities (range [0,1])
   scales: (N,3) tensor of Gaussian scales
   colors: (N,3) tensor of Gaussian RGB colors
   lx,ux,ly,uy,lz,uz: floats defining the world space box bounds (use arg `mask` for boolean tensor of segmented Gaussian masks)
   mask: (N,) boolean tensor, True for Gaussians to be edited (if None, default to world space bounding box with `lx,ux,ly,uy,lz,uz`)
   debug (optional): boolean for logging additional debugging information
   gsplat_path: path to nerfstudio model (eg. outputs/<model name>/splatfacto/<date>)

   Returns:
   new_means: (N,3) tensor in ckpt-space after deleted Gaussians
   quats_clone: (N,4) tensor of Gaussian orientations (w,x,y,z) in ckpt coords after deleted Gaussians
   opacities_clone: (N,1) tensor of Gaussian opacities after deleted Gaussians
   scales_clone: (N,3) tensor of Gaussian scales after deleted Gaussians
   colors_clone: (N,3) tensor of Gaussian RGB colors after deleted Gaussians
   """
   # 1) ckpt → world
   world = ckpt2world_xyz_batch(means, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")  # (N,3)
   
   # 2) build mask in world coordinates
   # NOTE:Either using bbox defined by user or 
   # (mask-id so that user does not need to track bbox)
   if mask is None:
      mask = (
         (world[:,0] >= lx) & (world[:,0] <= ux) &
         (world[:,1] >= ly) & (world[:,1] <= uy) &
         (world[:,2] >= lz) & (world[:,2] <= uz)
      )  # (N,)
   count = int(mask.sum().item())
   if debug:
      print("Gaussian within this bbox: ", count)
 
   # 3) shift along +X where mask is True
   world_shifted = world.clone()
   world_shifted = world_shifted[~mask]

   quats_clone = quats.clone()
   quats_clone = quats_clone[~mask]

   opacities_clone = opacities.clone()
   opacities_clone = opacities_clone[~mask]

   scales_clone = scales.clone()
   scales_clone = scales_clone[~mask]

   colors_clone = colors.clone()
   colors_clone = colors_clone[~mask]

   # 4) world → ckpt
   new_means = world2ckpt_xyz_batch(world_shifted, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")  # (N,3)

   return new_means, quats_clone, \
      opacities_clone, scales_clone, colors_clone

def duplicate(
    means: torch.Tensor,
    quats, opacities, scales, colors,
    tx=0, ty=0, tz=0,
    global_pos=None,
    yaw=0, pitch=0, roll=0,
    ux=0, lx=0,
    uy=0, ly=0,
    uz=0, lz=0,
    mask=None, debug=False, gsplat_path="outputs/multi_gates/splatfacto/2025-05-27_140811"
):
   """
   Return the new tensors and masks of gsplat with duplicated object defined by mask or bounding box
   
   means: (N,3) tensor in ckpt coords
   quats: (N,4) tensor of Gaussian orientations (w,x,y,z) in ckpt coords
   opacities: (N,1) tensor of Gaussian opacities (range [0,1])
   scales: (N,3) tensor of Gaussian scales
   colors: (N,3) tensor of Gaussian RGB colors
   tx,ty,tz: translation about world X,Y,Z axes of duplicated object (use `global_pos` for pose input rather than translation input)
   global_pos: (1,3) tensor of desired global pose of duplicated object in world coordinates (if None, default to `tx,ty,tz` translation)
   yaw,pitch,roll: rotation angles (radians) about world Z,Y,X axes of duplicated object
   lx,ux,ly,uy,lz,uz: floats defining the world space box bounds (use arg `mask` for segmented Gaussian masks)
   mask: (N,) boolean tensor, True for Gaussians to be edited (if None, default to world space bounding box with `lx,ux,ly,uy,lz,uz`)
   debug (optional): boolean for logging additional debugging information
   gsplat_path: path to nerfstudio model (eg. outputs/<model name>/splatfacto/<date>)

   Returns:
   new_means: (N,3) tensor in ckpt-space with duplicated object
   quats: (N,4) tensor of Gaussian orientations (w,x,y,z) in ckpt coords with duplicated object
   opacities: (N,1) tensor of Gaussian opacities with duplicated object
   scales: (N,3) tensor of Gaussian scales with duplicated object
   colors: (N,3) tensor of Gaussian RGB colors with duplicated object
   new_mask: (N,1) tensor of boolean mask of newly duplicated object
   """
   # 1) ckpt → world
   world = ckpt2world_xyz_batch(means, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")  # (N,3)
   
   # 2) build mask in world coordinates
   # NOTE:Either using bbox defined by user or 
   # (mask-id so that user does not need to track bbox)
   if mask is None:
      mask = (
         (world[:,0] >= lx) & (world[:,0] <= ux) &
         (world[:,1] >= ly) & (world[:,1] <= uy) &
         (world[:,2] >= lz) & (world[:,2] <= uz)
      )  # (N,)
   count = int(mask.sum().item())
   

   new_means = world[mask].clone()
   new_quats = quats[mask].clone()

   # Note that rotate() takes in ckpt space means
   new_mask = torch.ones(count, dtype=torch.bool, device=new_means.device)
   new_means, new_quats = rotate(means[mask].clone(), new_quats,
                                    yaw=yaw, pitch=pitch, roll=roll, mask=new_mask,
                                    gsplat_path=gsplat_path)
   # rotate() also return means in ckpt space so we need to do some conversion first
   new_means = ckpt2world_xyz_batch(new_means, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")  # (N,3)

   if global_pos is not None:
      cur_center =  new_means[new_mask, :].mean(dim=0)
      device = cur_center.device
      shift = torch.tensor(global_pos, device=device) - cur_center
      new_means[new_mask] += shift
   else:
      new_means[:, 0] += tx
      new_means[:, 1] += ty
      new_means[:, 2] += tz
 
   # 3) shift along +X where mask is True
   world_shifted = world.clone()
   world_shifted = torch.cat([world_shifted, new_means], dim=0)

   if debug:
      print("Gaussian within this bbox: ", count)
      center_point = new_means[:, :].mean(dim=0)
      print(f"New Object Center: {[round(p, 1) for p in center_point.tolist()]}")

   quats = torch.cat([quats, new_quats], dim=0)
   opacities = torch.cat([opacities, opacities[mask].clone()], dim=0)
   scales = torch.cat([scales, scales[mask].clone()], dim=0)
   colors = torch.cat([colors, colors[mask].clone()], dim=0)

   # 4) world → ckpt
   new_means = world2ckpt_xyz_batch(world_shifted, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")

   return new_means, quats, opacities, scales, colors, new_mask


def add(
    means: torch.Tensor,
    quats, opacities, scales, colors,
    pickle_file, #[means in world coordinates, quats, opacities, scales, colors]
    global_pos=None,
    yaw=0, pitch=0, roll=0,
    debug=False, customized_scale=1,
    gsplat_path="outputs/multi_gates/splatfacto/2025-05-27_140811"
):
   """
   Return the new tensor of Gaussians (and mask) with added object from pickle file
   
   means: (N,3) tensor in ckpt coords
   quats: (N,4) tensor of Gaussian orientations (w,x,y,z) in ckpt coords
   opacities: (N,1) tensor of Gaussian opacities (range [0,1])
   scales: (N,3) tensor of Gaussian scales
   colors: (N,3) tensor of Gaussian RGB colors
   pickle_file: path to pickle file of Gaussian object
   global_pos: (1,3) tensor of desired global pose to place object in world coordinates (if None, default to pose from `pickle_file`)
   yaw,pitch,roll: rotation angles (radians) of added object about world Z,Y,X axes
   debug (optional): boolean for logging additional debugging information
   customized_scale (optional): float value to scale the added Gaussian object
   gsplat_path: path to nerfstudio model (eg. outputs/<model name>/splatfacto/<date>)

   Returns:
   new_means: (N,3) tensor in ckpt-space with added object
   quats: (N,4) tensor of Gaussian orientations (w,x,y,z) in ckpt coords with added object
   opacities: (N,1) tensor of Gaussian opacities with added object
   scales: (N,3) tensor of Gaussian scales with added object
   colors: (N,3) tensor of Gaussian RGB colors with added object
   new_mask: (N,1) tensor of boolean mask of newly added object
   """
   with open(pickle_file, 'rb') as f:
      world_space_means, world_space_quats, new_opacities, new_scales, new_colors = pickle.load(f)

   # 1) ckpt → world
   world = ckpt2world_xyz_batch(means, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")  # (N,3)
   
   new_means = world_space_means.clone()
   # new_quats = new_quats.clone()
   new_quats = batch_rotm_world_to_quat_ckpt(world_space_quats, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")

   # Note that rotate() takes in ckpt space means
   new_mask = torch.ones(world_space_means.size(0), dtype=torch.bool, device=new_means.device)
   new_means = world2ckpt_xyz_batch(new_means, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")
   new_means, new_quats = rotate(new_means, new_quats,
                                    yaw=yaw, pitch=pitch, roll=roll, mask=new_mask,
                                    gsplat_path=gsplat_path
                                 )
   # rotate() also return means in ckpt space so we need to do some conversion first
   new_means = ckpt2world_xyz_batch(new_means, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")
   
   if global_pos is not None:
      cur_center =  new_means[new_mask, :].mean(dim=0)
      device = cur_center.device
      shift = torch.tensor(global_pos, device=device) - cur_center
      new_means[new_mask] += shift
      # print(cur_center, shift)
 
   # new_means, new_scales = scale_func(new_means, new_scales, mask=new_mask, scale=2)
   world_shifted = world.clone()
   world_shifted = torch.cat([world_shifted, new_means], dim=0)

   if debug:
      center_point = new_means[:, :].mean(dim=0)
      print(f"New Gate Center: {[round(p, 1) for p in center_point.tolist()]}")

   new_mask = torch.cat([torch.zeros(quats.size(0),dtype=torch.bool, device=device), new_mask], dim=0)
   quats = torch.cat([quats, new_quats], dim=0)
   opacities = torch.cat([opacities, new_opacities], dim=0)
   scales = torch.cat([scales, new_scales*customized_scale], dim=0)
   colors = torch.cat([colors, new_colors], dim=0)

   # 4) world → ckpt
   new_means = world2ckpt_xyz_batch(world_shifted, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")

   return new_means, quats, opacities, scales, colors, new_mask

def scale_func(
    means: torch.Tensor,
    scales,
    scale=0.9,
    ux=0, lx=0,
    uy=0, ly=0,
    uz=0, lz=0,
    mask=None, debug=False,
    gsplat_path="outputs/multi_gates/splatfacto/2025-05-27_140811"
):
   """
   Return the new tensor of Gaussians with the masked or bounding boxed regions of Gaussians scaled by factor
   
   means: (N,3) tensor in ckpt coords
   scales: (N,3) tensor of Gaussian scales
   scale: float value to scale the masked Gaussians
   lx,ux,ly,uy,lz,uz: floats defining the world space box bounds (use arg `mask` for segmented Gaussian masks)
   mask: (N,) boolean tensor, True for Gaussians to be edited (if None, default to world space bounding box with `lx,ux,ly,uy,lz,uz`)
   debug (optional): boolean for logging additional debugging information
   gsplat_path: path to nerfstudio model (eg. outputs/<model name>/splatfacto/<date>)

   Returns:
   new_means: (N,3) tensor in ckpt-space with new scaled Gaussians
   new_scales: (N,3) tensor of new Gaussian scales of masked Gaussians
   """
   # 1) ckpt → world
   world = ckpt2world_xyz_batch(means, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")  # (N,3)
   
   # 2) build mask in world coordinates
   # NOTE:Either using bbox defined by user or 
   # (mask-id so that user does not need to track bbox)
   if mask is None:
      mask = (
         (world[:,0] >= lx) & (world[:,0] <= ux) &
         (world[:,1] >= ly) & (world[:,1] <= uy) &
         (world[:,2] >= lz) & (world[:,2] <= uz)
      )  # (N,)
   count = int(mask.sum().item())
   

   # 3) shift where mask is True
   world_shifted = world.clone()

   center_point = world[mask].mean(dim=0)
   # Reposition each Gaussian center so that they scale about 'centroid'
   #     NewCenter = centroid + s * (OldCenter - centroid)
   world_shifted[mask] = center_point + scale * (world[mask] - center_point)

   new_scales = scales.clone()
   new_scales[mask] = 0.85 * scales[mask]

   if debug:
      print("Gaussian within this bbox: ", count)
      center_point = world_shifted[mask, :].mean(dim=0)
      print(f"Scaled Object Center: {[round(p, 1) for p in center_point.tolist()]}")

   # 4) world → ckpt
   new_means = world2ckpt_xyz_batch(world_shifted, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")

   return new_means, new_scales

def translate(
    means: torch.Tensor,
    tx=0, ty=0, tz=0, 
    global_pos=None,
    ux=0, lx=0,
    uy=0, ly=0,
    uz=0, lz=0,
    mask=None, debug=False,
    gsplat_path="outputs/multi_gates/splatfacto/2025-05-27_140811",
    record_mask_pickle_file=None
):
   """
   Return the new tensor of Gaussians (and mask) with added object from pickle file
   
   means: (N,3) tensor in ckpt coords
   tx,ty,tz: translation about world X,Y,Z axes of duplicated object (use `global_pos` for pose input rather than translation input)
   global_pos: (1,3) tensor of desired global pose of duplicated object in world coordinates (if None, default to `tx,ty,tz` translation)
   lx,ux,ly,uy,lz,uz: floats defining the world space box bounds (use arg `mask` for segmented Gaussian masks)
   mask: (N,) boolean tensor, True for Gaussians to be edited (if None, default to world space bounding box with `lx,ux,ly,uy,lz,uz`)
   debug (optional): boolean for logging additional debugging information
   gsplat_path: path to nerfstudio model (eg. outputs/<model name>/splatfacto/<date>)
   record_mask_pickle_file (optional): path to save new (N,) mask of translated object

   Returns:
   new_means: (N,3) tensor in ckpt-space with translated object
   mask: (N,) boolean tensor of translated Gaussians
   world_space_means: (N,3) tensor in world-space with translated object
   """
   # 1) ckpt → world
   world = ckpt2world_xyz_batch(means, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")  # (N,3)
   
   # 2) build mask in world coordinates
   # NOTE:Either using bbox defined by user or 
   # (mask-id so that user does not need to track bbox)
   if mask is None:
      mask = (
         (world[:,0] >= lx) & (world[:,0] <= ux) &
         (world[:,1] >= ly) & (world[:,1] <= uy) &
         (world[:,2] >= lz) & (world[:,2] <= uz)
      )  # (N,)
   count = int(mask.sum().item())

   if record_mask_pickle_file is not None:
      with open(record_mask_pickle_file, 'wb') as f:
         pickle.dump(mask, f, protocol=pickle.HIGHEST_PROTOCOL)


   # 3) shift where mask is True
   world_shifted = world.clone()

   if global_pos is not None:
      cur_center =  world_shifted[mask, :].mean(dim=0)
      device = cur_center.device
      shift = torch.tensor(global_pos, device=device) - cur_center
      world_shifted[mask] += shift
   else:
      world_shifted[mask, 0] += tx  # only moves masked points
      world_shifted[mask, 1] += ty  # only moves masked points
      world_shifted[mask, 2] += tz  # only moves masked points

   if debug:
      print("Gaussian within this bbox: ", count)
      center_point = world_shifted[mask, :].mean(dim=0)
      print(f"Translated Object Center: {[round(p, 1) for p in center_point.tolist()]}")

   # 4) world → ckpt
   new_means = world2ckpt_xyz_batch(world_shifted, dataparser_json_path=f"{gsplat_path}/dataparser_transforms.json")

   return new_means, mask, world_shifted[mask]


if __name__ == "__main__":
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(device)

   script_dir = os.path.dirname(os.path.realpath(__file__))
   output_folder = os.path.join(script_dir, '.')
   gsplat_path = "outputs/multi_gates/splatfacto/2025-05-27_140811"
   gsplat_path = "outputs/ellipse/splatfacto/2025-06-10_160003"
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

   # # # Select Tall Upper Gate
   # ux, lx = 4.5, 0.5
   # uy, ly = -16, -17
   # uz, lz = 2.1, -0.2
   ############ Test Translate ##############
   # with open('objects_IDs/base_gate_id_in_big_39999.pkl', 'rb') as f:
   #    mask = pickle.load(f)      # returns the same torch.Tensor object

   # means, quats, opacities, scales, colors, new_mask = duplicate(
   #    means, quats, opacities, scales, colors, \
   #    tx=2, tz =1,
   #    yaw=0, pitch=0, roll=0, \
   #    mask=mask, debug=True
   # )
   # mask = torch.cat([mask, torch.zeros(new_mask.size(0), dtype=torch.bool, device=mask.device)], dim=0)

   # s = time.time()
   # # means, scales = scale_func(
   # #    means, scales, mask=mask, scale=2, debug=True
   # # )
   # means  = translate(
   #    means, ux, lx, uy, ly, uz, lz,
   #    tx=2, tz=2,
   #    debug=True, gsplat_path=gsplat_path
   # )

   means, quats, opacities, scales, colors, new_mask = \
      add(means, quats, opacities, scales, colors, yaw=1.57, customized_scale=1.275,
          gsplat_path=gsplat_path,
          debug=True, pickle_file='objects_IDs/drone_gate.pkl', global_pos=[3, -20, 3])
   # print(means.shape)
   # print(f"Shift Time: {round(time.time()-s, 4)}s")

   for _ in range(1):
      pose = np.array([0, -8, 1, 1.57, 0, 0]) # (0, 0, 0) id=17 marker
      pose = np.array([6, -8, 1, 1.57, 0, 0]) #  id=10 marker
      pose = np.array([3, -8, 3, 1.57, 0.2, 0]) # center orange gate
      pose = np.array([0.3, 5, 1, 1.57, 0, 0]) # id=147 marker
      pose = np.array([6, 5, 3, 1.57, 0.2, 0]) # id=185 marker
      pose = np.array([2, -16.6, 0.9, 3.14, 0, 0]) # shorter gate
      pose = np.array([3, -22, 3, 1.57, 0, 0]) # shorter gate
      render_single_frame(pose, means, quats, opacities, scales, colors, device,
               gsplat_path=gsplat_path,
               # width = 1080, height = 1920, fx = 1633.1577699555364,
               # fy = 1638.0761006367832, cx = 540.0375363697227, cy = 959.7540360343551,
            )
   # ############ End Test Translate ##############

   # ############ Test Delete ##############
   # s = time.time()
   # means, quats, opacities, scales, colors, = delete(
   #    means, quats, opacities, scales, colors, ux, lx, uy, ly, uz, lz
   # )
   # print(f"Delete Time: {round(time.time()-s, 4)}s")

   # for _ in range(1):
   #    # pose = np.array([3, -30, 1.5, 1.57, 0, 0])
   #    pose = np.array([0, -8, 1, 1.57, 0, 0])
   #    # pose = np.array([1.5, -25, 1, 1.57, 0, 0]) # For group meeting presentation
   #    render_single_frame(pose, means, quats, opacities, scales, colors, device)
   # ############ End Test Delete ##############



   ############ Test Duplicate with Rotation & Translate ##############
   # s = time.time()
   # means, quats, opacities, scales, colors, new_mask = duplicate(
   #    means, quats, opacities, scales, colors, \
   #       ux, lx, uy, ly, uz, lz, \
   #       tx = -3, ty = 1, tz = 1.5, yaw=0.5, pitch=0, roll=0
   # )
   # print(f"Duplicate Time: {round(time.time()-s, 4)}s")

   # for _ in range(1):
   #    pose = np.array([1.5, -25, 1, 1.57, 0, 0])
   #    pose = np.array([3, -30, 3.5, 1.57, 0, 0]) # Front View
   #    render_single_frame(pose, means, quats, opacities, scales, colors, device)
   ############ End Test Duplicate with Translate ##############


   # ############ Test Rotate ##############
   # s = time.time()
   # means, quats = rotate(
   #    means, quats, ux, lx, uy, ly, uz, lz, -0.8, 0, 0
   # )
   # print(f"Rotate Time: {round(time.time()-s, 4)}s")

   # for _ in range(1):
   #    pose = np.array([10, -17, 1, 3.14, 0, 0]) # From right
   #    pose = np.array([3, -30, 3.5, 1.57, 0, 0]) # Front View
   #    # pose = np.array([-5, -17, 1, 0, 0, 0]) # From left
   #    pose = np.array([1.5, -25, 1, 1.57, 0, 0]) # For group meeting presentation
   #    render_single_frame(pose, means, quats, opacities, scales, colors, device)
   # ############ End Test Rotate ##############



   ############ Test Moving Gate and Camera Gate ############
   # # NOTE: Scenario 1: Gate moves in triangle pattern and camera is still
   # for idx in range(15):
   #    s = time.time()
   #    means = translate(
   #       means, ux, lx, uy, ly, uz, lz, tx=0.3, tz=0.1
   #    )
   #    ux += 0.3
   #    lx += 0.3
   #    uz += 0.1
   #    lz += 0.1
   #    print(f"Shift Time: {round(time.time()-s, 4)}s")

   #    for _ in range(1):
   #       pose = np.array([3, -30, 2.5, 1.57, 0, 0])
   #       img = render_single_frame(pose, means, quats, opacities, scales, colors, device,
   #          scale=1.5
   #       )
   #    cv2.imwrite(f'trajectory_images/{idx:03d}.png', img)
   
   # for idx in range(30):
   #    s = time.time()
   #    means = translate(
   #       means, ux, lx, uy, ly, uz, lz, tx=-0.3
   #    )
   #    ux += -0.3
   #    lx += -0.3
   #    print(f"Shift Time: {round(time.time()-s, 4)}s")

   #    for _ in range(1):
   #       pose = np.array([3, -30, 2.5, 1.57, 0, 0])
   #       img = render_single_frame(pose, means, quats, opacities, scales, colors, device,
   #          scale=1.5
   #       )
   #    cv2.imwrite(f'trajectory_images/{idx+15:03d}.png', img)
   
   # for idx in range(15):
   #    s = time.time()
   #    means = translate(
   #       means, ux, lx, uy, ly, uz, lz, tx=0.3, tz=-0.1
   #    )
   #    ux += 0.3
   #    lx += 0.3
   #    uz += -0.1
   #    lz += -0.1
   #    print(f"Shift Time: {round(time.time()-s, 4)}s")

   #    for _ in range(1):
   #       pose = np.array([3, -30, 2.5, 1.57, 0, 0])
   #       img = render_single_frame(pose, means, quats, opacities, scales, colors, device,
   #          scale=1.5
   #       )
   #    cv2.imwrite(f'trajectory_images/{idx+45:03d}.png', img)
   
   # # NOTE: Scenario 2: Gate moves in x-axis pattern and camera is also moving
   # dt = 0.05
   # direction = -1
   # for idx in range(15):
   #    s = time.time()
   #    means = translate(
   #       means, ux, lx, uy, ly, uz, lz, tx=0.3
   #    )
   #    ux += 0.3
   #    lx += 0.3
   #    print(f"Shift Time: {round(time.time()-s, 4)}s")

   #    for _ in range(1):
   #       pose = np.array([3-idx *0.5, -30+idx, 2.5, 1.57-idx*1.57/15, 0, 0])
   #       img = render_single_frame(pose, means, quats, opacities, scales, colors, device,
   #                scale=1.5
   #                )
   #    cv2.imwrite(f'trajectory_images/{idx:03d}.png', img)

   # for idx in range(30):
   #    s = time.time()
   #    means = translate(
   #       means, ux, lx, uy, ly, uz, lz, tx=-0.3
   #    )
   #    ux += -0.3
   #    lx += -0.3
   #    print(f"Shift Time: {round(time.time()-s, 4)}s")

   #    for _ in range(1):
   #       dt=idx
   #       if idx >=15:
   #          dt = 15
   #       pose = np.array([0-dt*0.5, -15+dt, 2.5, -dt*1/15, 0, 0])
   #       img = render_single_frame(pose, means, quats, opacities, scales, colors, device,
   #                scale=1.5, visualize=False
   #                )
   #    cv2.imwrite(f'trajectory_images/{idx+15:03d}.png', img)

   # for idx in range(30):
   #    s = time.time()
   #    means = translate(
   #       means, ux, lx, uy, ly, uz, lz, tx=0.3
   #    )
   #    ux += 0.3
   #    lx += 0.3
   #    print(f"Shift Time: {round(time.time()-s, 4)}s")

   #    for _ in range(1):
   #       pose = np.array([0-dt*0.5, -15+dt, 2.5, -dt*1/15, 0, 0])
   #       img = render_single_frame(pose, means, quats, opacities, scales, colors, device,
   #                scale=1.5, visualize=False
   #                )
   #    cv2.imwrite(f'trajectory_images/{idx+45:03d}.png', img)

   # for idx in range(30):
   #    s = time.time()
   #    means = translate(
   #       means, ux, lx, uy, ly, uz, lz, tx=-0.3
   #    )
   #    ux += -0.3
   #    lx += -0.3
   #    print(f"Shift Time: {round(time.time()-s, 4)}s")

   #    for _ in range(1):
   #       pose = np.array([0-dt*0.5, -15+dt, 2.5, -dt*1/15, 0, 0])
   #       img = render_single_frame(pose, means, quats, opacities, scales, colors, device,
   #                scale=1.5, visualize=False
   #                )
   #    cv2.imwrite(f'trajectory_images/{idx+75:03d}.png', img)


   ############ End Test Moving Gate and Camera Gate ############

   ############ Convert from raw scene to One Gate Scene ############
   # # # Select Tall Full Gate
   # # ux, lx = 4.5, 0.8
   # # uy, ly = -16, -17
   # # uz, lz = 2.1, -1.25

   # # # Select Tall Upper Gate
   # # ux, lx = 4.5, 0.8
   # # uy, ly = -16, -17
   # # uz, lz = 2.1, -0.2

   # # Select Tall Lower Gate
   # ux, lx = 4.5, 0.5
   # uy, ly = -16, -17
   # uz, lz = -0.19, -1.35
   # means, quats, opacities, scales, colors, = delete(
   #    means, quats, opacities, scales, colors, ux, lx, uy, ly, uz, lz,
   #    gsplat_path=gsplat_path
   # )

   # # Select (0, 0, 0) id=17 Marker
   # ux, lx = 1, -1
   # uy, ly = 0.5, -0.5
   # uz, lz = 0.65, -1.3
   # means, quats, opacities, scales, colors, = delete(
   #    means, quats, opacities, scales, colors, ux, lx, uy, ly, uz, lz,
   #    gsplat_path=gsplat_path
   # )

   # # Select id=10 Marker
   # ux, lx = 6.2+1.5, 6.2-1.5
   # uy, ly = 0.9, -0.9
   # uz, lz = 0.85, -1.35
   # means, quats, opacities, scales, colors, = delete(
   #    means, quats, opacities, scales, colors, ux, lx, uy, ly, uz, lz,
   #    gsplat_path=gsplat_path
   # )

   # # # Select center orange gate
   # # ux, lx = 3+1, 3-1.2
   # # uy, ly = 1.5, -0.5
   # # uz, lz = 0, -1.5

   # # Select id=147 Marker
   # ux, lx = 0.3+1, 0.3-1
   # uy, ly = 13+0.5, 13-0.5
   # uz, lz = 0.9, -1.4
   # means, quats, opacities, scales, colors, = delete(
   #    means, quats, opacities, scales, colors, ux, lx, uy, ly, uz, lz,
   #    gsplat_path=gsplat_path
   # )

   # # Select id=85 Marker
   # ux, lx = 6+1, 6-1
   # uy, ly = 13+0.5, 13-0.5
   # uz, lz = 0.9, -1.2
   # means, quats, opacities, scales, colors, = delete(
   #    means, quats, opacities, scales, colors, ux, lx, uy, ly, uz, lz,
   #    gsplat_path=gsplat_path
   # )

   # # Select shorter Gate
   # ux, lx = 4.5, 0.5
   # uy, ly = -6, -8
   # uz, lz = 0.9, -1.3
   # means, quats, opacities, scales, colors, = delete(
   #    means, quats, opacities, scales, colors, ux, lx, uy, ly, uz, lz,
   #    gsplat_path=gsplat_path
   # )
   # for _ in range(1):

   #    pose = np.array([2, -25, 3, 1.57, 0, 0]) # shorter gate
   #    render_single_frame(pose, means, quats, opacities, scales, colors, device,
   #             gsplat_path=gsplat_path
   #             # width = 1080, height = 1920, fx = 1633.1577699555364,
   #             # fy = 1638.0761006367832, cx = 540.0375363697227, cy = 959.7540360343551,
   #          )
   ############ EndConvert from raw scene to One Gate Scene ############

   ### Save new gsplat to a ckpt
   res['pipeline']['_model.gauss_params.means'] = means
   res['pipeline']['_model.gauss_params.quats'] = quats
   res['pipeline']['_model.gauss_params.opacities'] = opacities
   res['pipeline']['_model.gauss_params.scales'] = scales

   dc = colors[:, 0, :]
   rest = colors[:, 1:, :]
   res['pipeline']['_model.gauss_params.features_dc'] = dc
   res['pipeline']['_model.gauss_params.features_rest'] = rest

   # Save to a new ckpt   
   # torch.save(res, f"{gsplat_path}/nerfstudio_models/step-000039999.ckpt")
