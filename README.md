# FalconGym-Chenxi

### (Optional) Download Scene Data
You may either use your existing Nerfstudio data or download the pre-reconstructed [Nerfstudio scenes](https://drive.google.com/drive/folders/1koY1TL30Bty2x0U6VpszKRgMXk61oTkG?usp=drive_link) and place them in the below dictionary structure.

```bash
~/FalconGym-Chenxi/outputs/${case_name}/${reconstruction_method}/${datatime}/...
```
### Construct 3DGS Scene with different place of gates
```bash
export case_name=uturn
python3 scripts/generate_3D_gsplat.py --config configs/${case_name}/3d.yaml
```

### Generate traj.json and samples.json
```bash
export case_name=uturn
python3 scripts/generate_traj.py --config configs/${case_name}/traj.yaml
```

### Rendering images with camera poses captured along a traj.son or samples.json
```bash
export case_name=uturn
python3 scripts/traj_render.py --config configs/${case_name}/render.yaml
```

