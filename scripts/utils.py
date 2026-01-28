import numpy as np
from scipy.spatial.transform import Rotation
import json 
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# print(device)

# NOTE: return flag
#   0: safely cross gate
#   1: collide / outside gate
#   2: not on gate plane
def check_reach_gate(goal_state, pose, gate_radius=1, threshold_to_gate=0.15):
    MGE, return_flag = 0, 0
    # NOTE: Return value: 0 reach gate, 1 miss gate, 2 enroute to gate
    gate_x, gate_y, gate_z, gate_yaw = goal_state[:4] # (x, y) is the point on the line
    x, y, z = pose[:3]  # (a, b) is the test point
    # Handle vertical line cases (theta = pi/2 or -pi/2)
    if np.isclose(gate_yaw, np.pi / 2, atol=1e-3) or np.isclose(gate_yaw, -np.pi / 2, atol=1e-3):
        # The line is vertical, so the distance is the difference in x-coordinates
        distance = np.abs(gate_y - y)
    elif np.isclose(gate_yaw, 0, atol=1e-3) or np.isclose(gate_yaw, -np.pi, atol=1e-3) or np.isclose(gate_yaw, np.pi, atol=1e-3):
        distance = np.abs(gate_x - x)
    else:
        # Slope of the orthogonal plane of gates
        m = np.tan(gate_yaw)
        m = -1/m # slope of the plane direction
        # Line equation in the form Ax + By + C = 0
        A = m
        B = -1
        C = gate_y - m * gate_x
        # Distance formula
        distance = abs(A * x + B * y + C) / np.sqrt(A**2 + B**2)
    # print(distance)
    # Check if point is on gate plane
    if np.isclose(distance, 0, atol=threshold_to_gate):
        # Outside Gate
        if np.abs(gate_z-z) > gate_radius or np.abs(gate_y-y) > gate_radius or np.abs(gate_x-x) > gate_radius:
            # print(m, gate_yaw, x, y, z, distance, gate_x, gate_y, gate_z)
            # print(round(np.sqrt((gate_z - z)**2 + (gate_y - y)**2 + (gate_x - x)**2) , 2))
            MGE = np.sqrt((gate_z - z)**2 + (gate_y - y)**2 + (gate_x - x)**2)
            return MGE, 1
        # Safe Cross Gate
        else:
            # print(round(np.sqrt((gate_z - z)**2 + (gate_y - y)**2 + (gate_x - x)**2) , 2))
            MGE = np.sqrt((gate_z - z)**2 + (gate_y - y)**2 + (gate_x - x)**2)
            return MGE, 0
    else:
        return None, 2 # not on gate plane

# NOTE: simple conversion
#############################

def world2ckpt_xyz(world_xyz, dataparser_json_path="outputs/multi_gates/splatfacto/2025-05-27_140811/dataparser_transforms.json"):
    """
    world_xyz : array-like of shape (3,)  → [x, y, z] in your real-world frame
    dataparser_json_path : str path to your dataparser_transforms.json
    
    Returns
    -------
    ckpt_xyz : np.ndarray shape (3,)  in the ckpt frame
    """
    x, y, z = world_xyz
    world_xyz = np.array([-y, -z, x])

    # 1) load the same 4×4 JSON transform & scale
    dp = json.load(open(dataparser_json_path, 'r'))
    Xform     = np.array(dp['transform'])   # shape (4,4)
    Xform = np.vstack([Xform, np.array([[0,0,0,1]])])
    scale_fac =       dp['scale']           # scalar

    # 2) “tmp” axis rotation from your world2ckpt pipeline
    tmp = Rotation.from_euler('zyx', [-np.pi/2, np.pi/2, 0]).as_matrix()
    
    # 3) apply tmp to the world position
    r = tmp @ np.asarray(world_xyz)   # rotated position
    
    # 4) Colmap-style row reorder + flip:
    #    world2ckpt did: 
    #      new = [ r[0], r[2], -r[1] ]
    r1 = np.array([ r[0],  r[2], -r[1] ])
    
    # 5) embed as homogeneous and apply the JSON transform
    vec = np.ones(4)
    vec[:3] = r1
    vec = Xform @ vec
    
    # 6) apply the scale factor to the translation
    vec[:3] *= scale_fac
    
    return vec[:3]


def ckpt2world_xyz(ckpt_xyz, dataparser_json_path="outputs/multi_gates/splatfacto/2025-05-27_140811/dataparser_transforms.json"):
    """
    Inverse of world2ckpt_xyz:
    
    ckpt_xyz : array-like of shape (3,) in the ckpt frame
    dataparser_json_path : str path to your dataparser_transforms.json
    
    Returns
    -------
    world_xyz : np.ndarray shape (3,)  → [x, y, z] in your real-world frame
    """
    # 1) load & invert the JSON transform & scale
    dp = json.load(open(dataparser_json_path, 'r'))
    Xform     = np.array(dp['transform'])
    Xform = np.vstack([Xform, np.array([[0,0,0,1]])])
    inv_Xform = np.linalg.inv(Xform)
    scale_fac =       dp['scale']

    # 2) un‐scale
    v = np.asarray(ckpt_xyz) / scale_fac

    # 3) un‐transform
    vec = np.ones(4)
    vec[:3] = v
    vec = inv_Xform @ vec

    # 4) undo the Colmap reorder + flip:
    #    we had r1 = [ r0, r2, -r1 ]
    #    so    r0 = v0,  r1 = -v2,  r2 = v1
    v1 = vec[:3]
    r = np.array([ v1[0], -v1[2], v1[1] ])

    # 5) undo the tmp rotation
    tmp = Rotation.from_euler('zyx', [-np.pi/2, np.pi/2, 0]).as_matrix()
    world_xyz = tmp.T @ r

    x, y, z = world_xyz
    return [z, -x, -y]


def world2ckpt_xyz_batch(
    world_xyz: torch.Tensor, 
    dataparser_json_path: str = "outputs/multi_gates/splatfacto/2025-05-27_140811/dataparser_transforms.json"
) -> torch.Tensor:
    """
    world_xyz: (N,3) in real-world coordinates
    returns:   (N,3) in ckpt coordinates
    """
    device = world_xyz.device
    N = world_xyz.shape[0]

    # 1) initial reorder: [-y, -z, x]
    w = world_xyz
    r0 = torch.stack([-w[:,1], -w[:,2], w[:,0]], dim=1)  # (N,3)

    # 2) load & build Xform4 + scale
    dp        = json.load(open(dataparser_json_path, 'r'))
    X_numpy   = np.array(dp['transform'])                # (3,4) or (4,4)
    if X_numpy.shape == (3,4):
        X_numpy = np.vstack([X_numpy, [0,0,0,1]])        # make it (4,4)
    X4        = torch.tensor(X_numpy, dtype=torch.float32, device=device)
    scale_fac = float(dp['scale'])

    # 3) tmp rotation
    tmp_np = Rotation.from_euler('zyx', [-np.pi/2, np.pi/2, 0]).as_matrix()
    tmp    = torch.tensor(tmp_np, dtype=torch.float32, device=device)  # (3,3)

    # 4) apply tmp: r = tmp @ r0ᵀ  →  (r0 @ tmpᵀ)
    r = r0 @ tmp.T  # (N,3)

    # 5) colmap reorder: [r0, r2, -r1]
    r1 = torch.stack([r[:,0], r[:,2], -r[:,1]], dim=1)  # (N,3)

    # 6) homogeneous + JSON‐transform + scale
    ones = torch.ones(N,1, dtype=torch.float32, device=device)
    vec  = torch.cat([r1, ones], dim=1) @ X4.T          # (N,4)
    vec[:, :3] *= scale_fac

    return vec[:, :3]  # (N,3)


def ckpt2world_xyz_batch(
    ckpt_xyz: torch.Tensor,
    dataparser_json_path: str = "outputs/multi_gates/splatfacto/2025-05-27_140811/dataparser_transforms.json"
) -> torch.Tensor:
    """
    ckpt_xyz: (N,3) in ckpt coordinates
    returns:  (N,3) in real-world coordinates
    """
    device = ckpt_xyz.device
    N = ckpt_xyz.shape[0]

    # 1) load & invert Xform + scale
    dp        = json.load(open(dataparser_json_path, 'r'))
    X_numpy   = np.array(dp['transform'])
    if X_numpy.shape == (3,4):
        X_numpy = np.vstack([X_numpy, [0,0,0,1]])
    X4        = torch.tensor(X_numpy, dtype=torch.float32, device=device)
    invX4     = torch.inverse(X4)
    scale_fac = float(dp['scale'])

    # 2) un‐scale & un‐transform
    v    = ckpt_xyz / scale_fac                      # (N,3)
    ones = torch.ones(N,1, dtype=torch.float32, device=device)
    vec  = torch.cat([v, ones], dim=1) @ invX4.T      # (N,4)
    v1   = vec[:, :3]                                # (N,3)

    # 3) undo colmap‐reorder: r = [v0, -v2, v1]
    r    = torch.stack([v1[:,0], -v1[:,2], v1[:,1]], dim=1)  # (N,3)

    # 4) undo tmp: world = tmpᵀ @ r  →  r @ tmp
    tmp_np = Rotation.from_euler('zyx', [-np.pi/2, np.pi/2, 0]).as_matrix()
    tmp    = torch.tensor(tmp_np, dtype=torch.float32, device=device)  # (3,3)
    world = r @ tmp  # (N,3)

    # 5) final reorder back to [z, -x, -y]
    x, y, z = world[:,0], world[:,1], world[:,2]
    out = torch.stack([ z, -x, -y ], dim=1)  # (N,3)

    return out

# ——— helper: quaternion (w,x,y,z) → rotation matrix ———
def quat_to_rotm(quats: torch.Tensor) -> torch.Tensor:
    """
    quats: (N,4) in (w, x, y, z) order
    returns R: (N,3,3)
    """
    w, x, y, z = quats.unbind(-1)
    # precompute
    xx = x * x; yy = y * y; zz = z * z
    ww = w * w
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z

    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)

    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)

    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz

    # stack into (N,3,3)
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


# ——— helper: rotation matrix → quaternion (w,x,y,z) ———
def rotm_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    R: (N,3,3)
    returns quats: (N,4) in (w,x,y,z)
    Uses scipy on CPU for numerical robustness.
    """
    R_np = R.detach().cpu().numpy()
    rots = Rotation.from_matrix(R_np)
    q_xyzw = rots.as_quat()              # scipy gives (x,y,z,w)
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1)
    return torch.from_numpy(q_wxyz).to(R.device).type(R.dtype)


# ——— main: ckpt-quat → world-rotm ———
def batch_quat_ckpt_to_world_matrix(
    quats_ckpt: torch.Tensor,
    dataparser_json_path="outputs/multi_gates/splatfacto/2025-05-27_140811/dataparser_transforms.json"
) -> torch.Tensor:
    """
    quats_ckpt: (N,4) tensor in (w,x,y,z) style, in ckpt (gsplat) space
    returns:   (N,3,3) rotation matrices in real-world coords
    """
    device = quats_ckpt.device
    dtype  = quats_ckpt.dtype

    # 1) load the JSON transform
    dp = json.load(open(dataparser_json_path, 'r'))
    T = torch.tensor(dp['transform'], dtype=dtype, device=device)[:3, :3]  # (3,3)

    # 2) the fixed 'tmp' rotation from your world2ckpt
    tmp = torch.tensor(
        Rotation.from_euler('zyx', [-np.pi/2, np.pi/2, 0]).as_matrix(),
        dtype=dtype, device=device
    )

    # 3) build R_ckpt from input quaternions
    R_ckpt = quat_to_rotm(quats_ckpt)  # (N,3,3)

    # 4) invert the pipeline: R_world = P1 @ D3 @ Tᵀ @ R_ckpt @ tmpᵀ

    # 4a) undo the JSON rotation T → multiply by its transpose
    R1 = T.t().unsqueeze(0) @ R_ckpt         # (N,3,3)

    # 4b) undo the row-flip D3 (row 2 * -1)
    R1[:, 2, :] *= -1

    # 4c) undo the row-permute P1: rows [0,2,1]
    R2 = R1[:, [0, 2, 1], :]

    # 4d) undo the 'tmp' on the right
    R_world = R2 @ tmp.t()

    return R_world


# ——— main: world-rotm → ckpt-quat ———
def batch_rotm_world_to_quat_ckpt(
    R_world: torch.Tensor,
    dataparser_json_path="outputs/multi_gates/splatfacto/2025-05-27_140811/dataparser_transforms.json"
) -> torch.Tensor:
    """
    R_world: (N,3,3) rotation in real-world coords
    returns: (N,4) quats in (w,x,y,z) style, in ckpt space
    """
    device = R_world.device
    dtype  = R_world.dtype

    # 1) load JSON transform
    dp = json.load(open(dataparser_json_path, 'r'))
    T = torch.tensor(dp['transform'], dtype=dtype, device=device)[:3, :3]

    # 2) same 'tmp'
    tmp = torch.tensor(
        Rotation.from_euler('zyx', [-np.pi/2, np.pi/2, 0]).as_matrix(),
        dtype=dtype, device=device
    )

    # 3) forward pipeline: R_ckpt = T @ D3 @ P1 @ (R_world @ tmp)
    R_tmp = R_world @ tmp

    # 3a) row-permute P1: [0,2,1]
    R1 = R_tmp[:, [0, 2, 1], :]

    # 3b) row-flip D3
    R1[:, 2, :] *= -1

    # 3c) apply JSON T
    R_ckpt = T.unsqueeze(0) @ R1

    # 4) convert back to quaternion
    quats_ckpt = rotm_to_quat(R_ckpt)  # (N,4) in (w,x,y,z)

    return quats_ckpt
#######################




# NOTE: deprecated version for full xyz, yaw, pitch, roll conversion
#############################################################
def get_viewmat(optimized_camera_to_world):
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

# from real-world coordinate to ckpt coordinates:
def world2ckpt(pose):
    # Step 1: Construct 4x4 matrix in real-world coordinate system
    # pose = np.array([2.5, -15, -0.5, 1.57, 0, 0])
    px, py, pz, yaw, pitch, roll = pose
    rot = Rotation.from_euler("ZYX", (yaw, pitch, roll))
    R_matrix = rot.as_matrix()

    viewmats = np.eye(4)
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

    multi_gate_gsplat_path = "outputs/multi_gates/splatfacto/2025-05-27_140811"
    with open(f"{multi_gate_gsplat_path}/dataparser_transforms.json", 'r') as f:
        dp_trans_info = json.load(f)
    transform = np.array(dp_trans_info['transform'])
    scale_factor = dp_trans_info['scale']
    viewmats = transform@viewmats
    viewmats[:3,3] *= scale_factor
    viewmats = viewmats[:3,:]
    if viewmats.ndim == 2:
        viewmats = np.expand_dims(viewmats, axis=0)
    viewmats = torch.FloatTensor( viewmats ).to(device)


    viewmats = get_viewmat(viewmats)#.to(device)

    # Option A: invert the homogeneous matrix, then read off the translation
    c2w   = torch.linalg.inv(viewmats[0])     # now camera-to-world in ckpt coords
    xyz   = c2w[:3, 3]                 # (x, y, z) of the camera in ckpt frame
    # # Option B: without a full inverse, use R_inv, t_inv directly
    # R_inv = view[:3, :3]               # this is R^T from your pipeline
    # t_inv = view[:3,  3]               # this is T_inv = -R^T T_world
    # xyz  = -R_inv.T @ t_inv           # same as inverting

    # 2) grab the 3×3 rotation
    rot = c2w[:3, :3]
    # 3) extract yaw, pitch, roll in ZYX order
    #    (i.e. yaw around Z, pitch around Y, roll around X)
    yaw, pitch, roll = Rotation.from_matrix(rot).as_euler('ZYX', degrees=False)

    return viewmats, xyz.tolist(), [yaw, pitch, roll]


def ckpt2world(viewmat, dataparser_json_path="outputs/multi_gates/splatfacto/2025-05-27_140811/dataparser_transforms.json"):
    """
    Invert your world2ckpt pipeline.
    viewmat:   (4x4) numpy array or torch tensor (world2camera matrix from ckpt coords)
    dataparser_json_path: path to dataparser_transforms.json (with 'transform' & 'scale')
    Returns:
      pose:  array [x, y, z, yaw, pitch, roll] in real-world coordinates
    """
    # --- 1) load the same transform & scale you used in world2ckpt
    dp = json.load(open(dataparser_json_path, 'r'))
    Xform     = np.array(dp['transform'])        # 4×4
    Xform = np.vstack([Xform, np.array([[0,0,0,1]])])
    scale_fac = dp['scale']                      # scalar

    # --- 2) bring viewmat into numpy, ensure shape (4,4)
    if hasattr(viewmat, 'cpu'):  # torch tensor?
        V = viewmat.detach().cpu().numpy()
    else:
        V = np.array(viewmat)
    assert V.shape == (4,4), "expected a single 4x4 matrix"

    # --- 3) invert get_viewmat (undo world→camera)
    C = np.diag([1, -1, -1])            # the same flip diag
    R_inv = V[:3, :3]
    t_inv = V[:3,  3 ]
    # R1 = original R_after_flip;   R1 = (R_orig @ C)
    # we have R_inv = R1^T →  R1 = (R_inv)^T
    R1 = R_inv.T
    # so R_orig = R1 @ C   since R1 = R_orig @ C
    R_orig = R1 @ C
    # and t_orig = - R_orig @ C @ t_inv
    t_orig = - R_orig @ (C @ t_inv)

    M2 = np.eye(4)
    M2[:3,:3] = R_orig
    M2[:3,  3] = t_orig

    # --- 4) undo scale & JSON-transform
    M2[:3, 3] /= scale_fac

    M1 = np.linalg.inv(Xform) @ M2

    # --- 5) undo Colmap‐style alignment
    #    world2ckpt did:
    #      a) M[:3,1:3] *= -1
    #      b) M = M[[0,2,1,3],:]
    #      c) M[2,:]   *= -1
    #
    #  Inverse is reverse order:
    M1[2, :]   *= -1
    M2p = np.zeros_like(M1)
    M2p[0, :]  = M1[0, :]
    M2p[1, :]  = M1[2, :]
    M2p[2, :]  = M1[1, :]
    M2p[3, :]  = M1[3, :]
    M2p[:3,1:3] *= -1
    # now M2p is the matrix just after your “tmp” rotation

    # --- 6) undo the tmp rotation on R only
    tmp = Rotation.from_euler('zyx', [-np.pi/2, np.pi/2, 0]).as_matrix()
    R_temp = M2p[:3,:3]
    R0     = R_temp @ tmp.T
    t0     =    M2p[:3, 3 ]

    # --- 7) extract Euler angles in ZYX order (yaw, pitch, roll)
    yaw, pitch, roll = Rotation.from_matrix(R0).as_euler('ZYX')

    return np.array([t0[0], t0[1], t0[2], yaw, pitch, roll])



#############################################################

if __name__ == "__main__":
    # say your JSON is at:
    json_path = "outputs/multi_gates/splatfacto/2025-05-27_140811/dataparser_transforms.json"

    # suppose you already did:
    pose = np.array([1.1, -17, 1.4, 1.57, 0, 0])
    view, xyz, rot = world2ckpt(pose)        # pick the (4×4) matrix
    print("ckpt coord: ",xyz, rot)
    view = view.squeeze(0)
    pose_rec = ckpt2world(view)
    print("Recovered world [x,y,z,yaw,pitch,roll] =", pose_rec)

    # convert a world point → ckpt frame
    w = [0.8, -16, -0.5]
    c = world2ckpt_xyz(w, json_path)
    print("ckpt coords -- only xyz:", c)

    # convert back → should recover the original
    w_rec = ckpt2world_xyz([0.154, 0.227, -0.015], json_path)
    # w_rec = ckpt2world_xyz(c, json_path)
    print("recovered world coords -- only xyz:", w_rec)

    ux, lx = 4.5, 0.8
    uy, ly =  -16,  -17
    uz, lz =  2.1, -0.6
    C = torch.tensor([[0.154, 0.227, -0.015]])

    W = ckpt2world_xyz_batch(C)

    mask = (
        (W[:,0] >= lx) & (W[:,0] <= ux) &
        (W[:,1] >= ly) & (W[:,1] <= uy) &
        (W[:,2] >= lz) & (W[:,2] <= uz)
    )  # (N,)
    count = int(mask.sum().item())
    print("HERE", count)
    print(C, W)

