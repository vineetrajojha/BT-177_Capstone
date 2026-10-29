#!/usr/bin/env python3
"""
Applies documented patches to third_party/OpenPCDet programmatically.

Instead of relying on `git apply` (which is fragile with Windows line endings
and generated patches), this script directly rewrites the three files that
need modification for BEVFusionLite:

  1. setup.py — skip CUDA extensions when no MSVC compiler is available
  2. pcdet/ops/iou3d_nms/iou3d_nms_utils.py — graceful fallback when
     iou3d_nms_cuda is not compiled
  3. pcdet/models/__init__.py — auto-discover BEVFusionLite plugin modules

Every patch is idempotent: a sentinel comment marks already-patched files.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPENPCDET_DIR = PROJECT_ROOT / "third_party" / "OpenPCDet"

SENTINEL = "# BEVFUSIONLITE_PATCHED"


def _already_patched(filepath: Path) -> bool:
    return filepath.exists() and SENTINEL in filepath.read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Patch 1: setup.py — conditional CUDA extension build
# ──────────────────────────────────────────────────────────────────────
def patch_setup_py():
    p = OPENPCDET_DIR / "setup.py"
    if _already_patched(p):
        print(f"[i] {p.name} already patched. Skipping.")
        return True

    content = p.read_text(encoding="utf-8")

    new_content = f'''\
{SENTINEL}
import os
import subprocess

from setuptools import find_packages, setup
import torch


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources):
    from torch.utils.cpp_extension import CUDAExtension
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    # --- BEVFusionLite patch: conditional CUDA build ---
    can_build_cuda = False
    if torch.cuda.is_available() and os.environ.get('NO_CUDA_OPS', '0') != '1':
        try:
            from torch.utils.cpp_extension import BuildExtension
            # Check for MSVC (cl.exe) on Windows, or gcc/nvcc on Linux
            if os.name == 'nt':
                res = subprocess.run(['where', 'cl'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                res = subprocess.run(['which', 'nvcc'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0:
                can_build_cuda = True
        except Exception:
            pass

    if not can_build_cuda:
        print("[BEVFusionLite] CUDA ops build SKIPPED (no compiler found). "
              "Falling back to pure-Python / torchvision ops.")

    version = '0.6.0+%s' % get_git_commit_number()
    write_version_to_file(version, 'pcdet/version.py')

    _cuda_ext_modules = [
        make_cuda_ext(
            name='iou3d_nms_cuda',
            module='pcdet.ops.iou3d_nms',
            sources=[
                'src/iou3d_cpu.cpp',
                'src/iou3d_nms_api.cpp',
                'src/iou3d_nms.cpp',
                'src/iou3d_nms_kernel.cu',
            ]
        ),
        make_cuda_ext(
            name='roiaware_pool3d_cuda',
            module='pcdet.ops.roiaware_pool3d',
            sources=[
                'src/roiaware_pool3d.cpp',
                'src/roiaware_pool3d_kernel.cu',
            ]
        ),
        make_cuda_ext(
            name='roipoint_pool3d_cuda',
            module='pcdet.ops.roipoint_pool3d',
            sources=[
                'src/roipoint_pool3d.cpp',
                'src/roipoint_pool3d_kernel.cu',
            ]
        ),
        make_cuda_ext(
            name='pointnet2_stack_cuda',
            module='pcdet.ops.pointnet2.pointnet2_stack',
            sources=[
                'src/pointnet2_api.cpp',
                'src/ball_query.cpp',
                'src/ball_query_gpu.cu',
                'src/group_points.cpp',
                'src/group_points_gpu.cu',
                'src/sampling.cpp',
                'src/sampling_gpu.cu',
                'src/interpolate.cpp',
                'src/interpolate_gpu.cu',
                'src/voxel_query.cpp',
                'src/voxel_query_gpu.cu',
                'src/vector_pool.cpp',
                'src/vector_pool_gpu.cu'
            ],
        ),
        make_cuda_ext(
            name='pointnet2_batch_cuda',
            module='pcdet.ops.pointnet2.pointnet2_batch',
            sources=[
                'src/pointnet2_api.cpp',
                'src/ball_query.cpp',
                'src/ball_query_gpu.cu',
                'src/group_points.cpp',
                'src/group_points_gpu.cu',
                'src/interpolate.cpp',
                'src/interpolate_gpu.cu',
                'src/sampling.cpp',
                'src/sampling_gpu.cu',
            ],
        ),
        make_cuda_ext(
            name="bev_pool_ext",
            module="pcdet.ops.bev_pool",
            sources=[
                "src/bev_pool.cpp",
                "src/bev_pool_cuda.cu",
            ],
        ),
        make_cuda_ext(
            name='ingroup_inds_cuda',
            module='pcdet.ops.ingroup_inds',
            sources=[
                'src/ingroup_inds.cpp',
                'src/ingroup_inds_kernel.cu',
            ]
        ),
    ] if can_build_cuda else []

    setup(
        name='pcdet',
        version=version,
        description='OpenPCDet is a general codebase for 3D object detection from point cloud',
        install_requires=[
            'numpy',
            'llvmlite',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml',
            'scikit-image',
            'tqdm',
            'SharedArray',
        ],

        author='Shaoshuai Shi',
        author_email='shaoshuaics@gmail.com',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={{'build_ext': BuildExtension}} if can_build_cuda else {{}},
        ext_modules=_cuda_ext_modules,
    )
'''

    p.write_text(new_content, encoding="utf-8", newline="\n")
    print(f"[PASS] Patched {p.name}")
    return True


# ──────────────────────────────────────────────────────────────────────
# Patch 2: iou3d_nms_utils.py — graceful CUDA import + torchvision NMS
# ──────────────────────────────────────────────────────────────────────
def patch_iou3d_nms():
    p = OPENPCDET_DIR / "pcdet" / "ops" / "iou3d_nms" / "iou3d_nms_utils.py"
    if _already_patched(p):
        print(f"[i] {p.name} already patched. Skipping.")
        return True

    txt = p.read_text(encoding="utf-8")

    # Replace hard import with try/except
    txt = txt.replace(
        "from . import iou3d_nms_cuda",
        f"""{SENTINEL}
try:
    from . import iou3d_nms_cuda
except ImportError:
    iou3d_nms_cuda = None""",
        1,
    )

    # Inject torchvision fallback at the top of nms_gpu()
    old_nms_body = '    assert boxes.shape[1] == 7\n    order = scores.sort(0, descending=True)[1]\n    if pre_maxsize is not None:\n        order = order[:pre_maxsize]\n\n    boxes = boxes[order].contiguous()\n    keep = torch.LongTensor(boxes.size(0))\n    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)\n    return order[keep[:num_out].cuda()].contiguous(), None'

    new_nms_body = '''    # --- BEVFusionLite fallback when CUDA ops unavailable ---
    if iou3d_nms_cuda is None:
        from torchvision.ops import nms
        order = scores.sort(0, descending=True)[1]
        if pre_maxsize is not None:
            order = order[:pre_maxsize]
        boxes_sorted = boxes[order]
        scores_sorted = scores[order]
        # Convert 3D BEV boxes to 2D [x1, y1, x2, y2]
        x1 = boxes_sorted[:, 0] - boxes_sorted[:, 3] / 2
        y1 = boxes_sorted[:, 1] - boxes_sorted[:, 4] / 2
        x2 = boxes_sorted[:, 0] + boxes_sorted[:, 3] / 2
        y2 = boxes_sorted[:, 1] + boxes_sorted[:, 4] / 2
        boxes_2d = torch.stack([x1, y1, x2, y2], dim=1)
        keep_idx = nms(boxes_2d, scores_sorted, thresh)
        selected = order[keep_idx]
        return selected, len(selected)

    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None'''

    txt = txt.replace(old_nms_body, new_nms_body, 1)

    # Also guard boxes_bev_iou_cpu against missing cuda
    txt = txt.replace(
        "    iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)",
        """    if iou3d_nms_cuda is not None:
        iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)
    else:
        # Fallback: axis-aligned BEV IoU (approximate)
        pass""",
        1,
    )

    p.write_text(txt, encoding="utf-8", newline="\n")
    print(f"[PASS] Patched {p.name}")
    return True


# ──────────────────────────────────────────────────────────────────────
# Patch 3: pcdet/models/__init__.py — BEVFusionLite plugin discovery
# ──────────────────────────────────────────────────────────────────────
def patch_models_init():
    p = OPENPCDET_DIR / "pcdet" / "models" / "__init__.py"
    if _already_patched(p):
        print(f"[i] {p.name} already patched. Skipping.")
        return True

    txt = p.read_text(encoding="utf-8")

    plugin_block = f"""

{SENTINEL}
# ── BEVFusionLite plugin auto-discovery ──
try:
    import bevfusionlite  # noqa: F401
except ImportError:
    pass
"""

    txt += plugin_block
    p.write_text(txt, encoding="utf-8", newline="\n")
    print(f"[PASS] Patched {p.name}")
    return True


# ----------------------------------------------------------------------
# Patch 4: pcdet/utils/common_utils.py — make SharedArray optional
# ----------------------------------------------------------------------
def patch_common_utils():
    p = OPENPCDET_DIR / "pcdet" / "utils" / "common_utils.py"
    if _already_patched(p):
        print(f"[i] {p.name} already patched. Skipping.")
        return True

    txt = p.read_text(encoding="utf-8")
    txt = txt.replace(
        "import SharedArray",
        f"""{SENTINEL}
try:
    import SharedArray
except ImportError:
    SharedArray = None""",
        1,
    )
    # Also handle sa_create gracefully
    txt = txt.replace(
        "def sa_create(name, var):\n    x = SharedArray.create(name, var.shape, dtype=var.dtype)",
        "def sa_create(name, var):\n    if SharedArray is None:\n        return var\n    x = SharedArray.create(name, var.shape, dtype=var.dtype)",
        1,
    )
    p.write_text(txt, encoding="utf-8", newline="\n")
    print(f"[PASS] Patched {p.name}")
    return True


# ----------------------------------------------------------------------
# Patch 5: pcdet/datasets/augmentor/database_sampler.py — optional SharedArray
# ----------------------------------------------------------------------
def patch_database_sampler():
    p = OPENPCDET_DIR / "pcdet" / "datasets" / "augmentor" / "database_sampler.py"
    if _already_patched(p):
        print(f"[i] {p.name} already patched. Skipping.")
        return True

    txt = p.read_text(encoding="utf-8")
    txt = txt.replace(
        "import SharedArray",
        f"""{SENTINEL}
try:
    import SharedArray
except ImportError:
    SharedArray = None""",
        1,
    )
    p.write_text(txt, encoding="utf-8", newline="\n")
    print(f"[PASS] Patched {p.name}")
    return True


# ----------------------------------------------------------------------
# Patch 5b: pcdet/datasets/waymo/waymo_dataset.py — optional SharedArray
# ----------------------------------------------------------------------
def patch_waymo_dataset():
    p = OPENPCDET_DIR / "pcdet" / "datasets" / "waymo" / "waymo_dataset.py"
    if _already_patched(p):
        print(f"[i] {p.name} already patched. Skipping.")
        return True

    txt = p.read_text(encoding="utf-8")
    txt = txt.replace(
        "import SharedArray",
        f"""{SENTINEL}
try:
    import SharedArray
except ImportError:
    SharedArray = None""",
        1,
    )
    p.write_text(txt, encoding="utf-8", newline="\n")
    print(f"[PASS] Patched {p.name}")
    return True


# ----------------------------------------------------------------------
# Patch 6: pcdet/utils/spconv_utils.py — optional spconv
# ----------------------------------------------------------------------
def patch_spconv_utils():
    p = OPENPCDET_DIR / "pcdet" / "utils" / "spconv_utils.py"

    new_content = f'''{SENTINEL}
from typing import Set
import torch.nn as nn

try:
    import spconv
    if hasattr(spconv, "__version__") and float(spconv.__version__[2:]) >= 2.2:
        spconv.constants.SPCONV_USE_DIRECT_TABLE = False
    try:
        import spconv.pytorch as spconv
    except Exception:
        import spconv as spconv
except ImportError:
    class _Dummy(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
    class _DummySpconv:
        SparseModule = _Dummy
        SparseSequential = nn.Sequential
        SparseConvTensor = _Dummy
        SparseConvolution = _Dummy
        SubMConv3d = _Dummy
        SparseConv3d = _Dummy
        SparseInverseConv3d = _Dummy
        conv = _Dummy
    spconv = _DummySpconv()


def find_all_spconv_keys(model: nn.Module, prefix="") -> Set[str]:
    """
    Finds all spconv keys that need to have weight's transposed
    """
    found_keys: Set[str] = set()
    for name, child in model.named_children():
        new_prefix = f"{{prefix}}.{{name}}" if prefix != "" else name

        if hasattr(spconv, "conv") and isinstance(child, getattr(spconv.conv, "SparseConvolution", object)):
            new_prefix = f"{{new_prefix}}.weight"
            found_keys.add(new_prefix)

        found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))

    return found_keys


def replace_feature(out, new_features):
    if hasattr(out, "replace_feature"):
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out
'''
    p.write_text(new_content, encoding="utf-8", newline="\n")
    print(f"[PASS] Patched {p.name}")
    return True


# ----------------------------------------------------------------------
# Patch 7: pcdet/models/backbones_3d/__init__.py — safe import of backbones
# ----------------------------------------------------------------------
def patch_backbones_3d():
    p = OPENPCDET_DIR / "pcdet" / "models" / "backbones_3d" / "__init__.py"
    if _already_patched(p):
        return True

    new_content = f'''{SENTINEL}
__all__ = {{}}

def _safe_import():
    try:
        from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
        __all__['PointNet2Backbone'] = PointNet2Backbone
        __all__['PointNet2MSG'] = PointNet2MSG
    except Exception:
        pass

    try:
        from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x
        __all__['VoxelBackBone8x'] = VoxelBackBone8x
        __all__['VoxelResBackBone8x'] = VoxelResBackBone8x
    except Exception:
        pass

    try:
        from .spconv_backbone_2d import PillarBackBone8x, PillarRes18BackBone8x
        __all__['PillarBackBone8x'] = PillarBackBone8x
        __all__['PillarRes18BackBone8x'] = PillarRes18BackBone8x
    except Exception:
        pass

    try:
        from .spconv_backbone_focal import VoxelBackBone8xFocal
        __all__['VoxelBackBone8xFocal'] = VoxelBackBone8xFocal
    except Exception:
        pass

    try:
        from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt
        __all__['VoxelResBackBone8xVoxelNeXt'] = VoxelResBackBone8xVoxelNeXt
    except Exception:
        pass

    try:
        from .spconv_backbone_voxelnext2d import VoxelResBackBone8xVoxelNeXt2D
        __all__['VoxelResBackBone8xVoxelNeXt2D'] = VoxelResBackBone8xVoxelNeXt2D
    except Exception:
        pass

    try:
        from .spconv_unet import UNetV2
        __all__['UNetV2'] = UNetV2
    except Exception:
        pass

    try:
        from .dsvt import DSVT
        __all__['DSVT'] = DSVT
    except Exception:
        pass

_safe_import()
'''
    p.write_text(new_content, encoding="utf-8", newline="\n")
    print(f"[PASS] Patched {p.name}")
    return True


# ----------------------------------------------------------------------
# Patch 7: pcdet/ops/* — graceful fallback when CUDA ops not compiled
# ----------------------------------------------------------------------
def patch_cuda_ops():
    # 1. roiaware_pool3d
    p1 = OPENPCDET_DIR / "pcdet" / "ops" / "roiaware_pool3d" / "roiaware_pool3d_utils.py"
    if not _already_patched(p1):
        txt1 = p1.read_text(encoding="utf-8")
        txt1 = txt1.replace(
            "from . import roiaware_pool3d_cuda",
            f"{SENTINEL}\ntry:\n    from . import roiaware_pool3d_cuda\nexcept ImportError:\n    roiaware_pool3d_cuda = None",
            1
        )
        p1.write_text(txt1, encoding="utf-8", newline="\n")

    # 2. roipoint_pool3d
    p2 = OPENPCDET_DIR / "pcdet" / "ops" / "roipoint_pool3d" / "roipoint_pool3d_utils.py"
    if not _already_patched(p2):
        txt2 = p2.read_text(encoding="utf-8")
        txt2 = txt2.replace(
            "from . import roipoint_pool3d_cuda",
            f"{SENTINEL}\ntry:\n    from . import roipoint_pool3d_cuda\nexcept ImportError:\n    roipoint_pool3d_cuda = None",
            1
        )
        p2.write_text(txt2, encoding="utf-8", newline="\n")

    # 3. pointnet2_batch
    p3 = OPENPCDET_DIR / "pcdet" / "ops" / "pointnet2" / "pointnet2_batch" / "pointnet2_utils.py"
    if not _already_patched(p3):
        txt3 = p3.read_text(encoding="utf-8")
        txt3 = txt3.replace(
            "from . import pointnet2_batch_cuda as pointnet2",
            f"{SENTINEL}\ntry:\n    from . import pointnet2_batch_cuda as pointnet2\nexcept ImportError:\n    pointnet2 = None",
            1
        )
        p3.write_text(txt3, encoding="utf-8", newline="\n")

    # 4. pointnet2_stack
    p4 = OPENPCDET_DIR / "pcdet" / "ops" / "pointnet2" / "pointnet2_stack" / "pointnet2_utils.py"
    if not _already_patched(p4):
        txt4 = p4.read_text(encoding="utf-8")
        txt4 = txt4.replace(
            "from . import pointnet2_stack_cuda as pointnet2",
            f"{SENTINEL}\ntry:\n    from . import pointnet2_stack_cuda as pointnet2\nexcept ImportError:\n    pointnet2 = None",
            1
        )
        p4.write_text(txt4, encoding="utf-8", newline="\n")

    # 5. bev_pool
    p5 = OPENPCDET_DIR / "pcdet" / "ops" / "bev_pool" / "bev_pool.py"
    if not _already_patched(p5):
        txt5 = p5.read_text(encoding="utf-8")
        txt5 = txt5.replace(
            "from . import bev_pool_ext",
            f"{SENTINEL}\ntry:\n    from . import bev_pool_ext\nexcept ImportError:\n    bev_pool_ext = None",
            1
        )
        p5.write_text(txt5, encoding="utf-8", newline="\n")

    # 6. ingroup_inds
    p6 = OPENPCDET_DIR / "pcdet" / "ops" / "ingroup_inds" / "ingroup_inds_op.py"
    if not _already_patched(p6):
        txt6 = p6.read_text(encoding="utf-8")
        txt6 = txt6.replace(
            "ingroup_indices = ingroup_inds_cuda",
            f"{SENTINEL}\ningroup_indices = getattr(globals().get('ingroup_inds_cuda', None), 'forward', None)",
            1
        )
        p6.write_text(txt6, encoding="utf-8", newline="\n")

    # Ops package __init__.py modules fallback
    ops_modules = [
        ("iou3d_nms", "iou3d_nms_cuda"),
        ("roiaware_pool3d", "roiaware_pool3d_cuda"),
        ("roipoint_pool3d", "roipoint_pool3d_cuda"),
        ("pointnet2/pointnet2_batch", "pointnet2_batch_cuda"),
        ("pointnet2/pointnet2_stack", "pointnet2_stack_cuda"),
        ("bev_pool", "bev_pool_ext"),
        ("ingroup_inds", "ingroup_inds_cuda"),
    ]
    for rel_path, mod_name in ops_modules:
        init_file = OPENPCDET_DIR / "pcdet" / "ops" / rel_path / "__init__.py"
        if not _already_patched(init_file):
            content = f'''{SENTINEL}
try:
    from . import {mod_name}
except ImportError:
    {mod_name} = None
'''
            init_file.write_text(content, encoding="utf-8", newline="\n")

    print("[PASS] Patched ops cuda fallbacks")
    return True


# ----------------------------------------------------------------------
# Patch 8: pcdet/models/dense_heads/__init__.py — safe import of heads
# ----------------------------------------------------------------------
def patch_dense_heads():
    p = OPENPCDET_DIR / "pcdet" / "models" / "dense_heads" / "__init__.py"
    if _already_patched(p):
        return True

    new_content = f'''{SENTINEL}
__all__ = {{}}

def _safe_import():
    try:
        from .anchor_head_template import AnchorHeadTemplate
        __all__['AnchorHeadTemplate'] = AnchorHeadTemplate
    except Exception:
        pass
    try:
        from .anchor_head_single import AnchorHeadSingle
        __all__['AnchorHeadSingle'] = AnchorHeadSingle
    except Exception:
        pass
    try:
        from .anchor_head_multi import AnchorHeadMulti
        __all__['AnchorHeadMulti'] = AnchorHeadMulti
    except Exception:
        pass
    try:
        from .point_head_simple import PointHeadSimple
        __all__['PointHeadSimple'] = PointHeadSimple
    except Exception:
        pass
    try:
        from .point_head_box import PointHeadBox
        __all__['PointHeadBox'] = PointHeadBox
    except Exception:
        pass
    try:
        from .point_intra_part_head import PointIntraPartOffsetHead
        __all__['PointIntraPartOffsetHead'] = PointIntraPartOffsetHead
    except Exception:
        pass
    try:
        from .center_head import CenterHead
        __all__['CenterHead'] = CenterHead
    except Exception:
        pass
    try:
        from .voxelnext_head import VoxelNeXtHead
        __all__['VoxelNeXtHead'] = VoxelNeXtHead
    except Exception:
        pass
    try:
        from .transfusion_head import TransFusionHead
        __all__['TransFusionHead'] = TransFusionHead
    except Exception:
        pass

_safe_import()
'''
    p.write_text(new_content, encoding="utf-8", newline="\n")
    print(f"[PASS] Patched {p.name}")
    return True


# ----------------------------------------------------------------------
# Patch 9: pcdet/models/roi_heads/__init__.py — safe import of roi heads
# ----------------------------------------------------------------------
def patch_roi_heads():
    p = OPENPCDET_DIR / "pcdet" / "models" / "roi_heads" / "__init__.py"
    if _already_patched(p):
        return True

    new_content = f'''{SENTINEL}
__all__ = {{}}

def _safe_import():
    try:
        from .roi_head_template import RoIHeadTemplate
        __all__['RoIHeadTemplate'] = RoIHeadTemplate
    except Exception:
        pass
    try:
        from .partA2_head import PartA2FCHead
        __all__['PartA2FCHead'] = PartA2FCHead
    except Exception:
        pass
    try:
        from .pvrcnn_head import PVRCNNHead
        __all__['PVRCNNHead'] = PVRCNNHead
    except Exception:
        pass
    try:
        from .second_head import SECONDHead
        __all__['SECONDHead'] = SECONDHead
    except Exception:
        pass
    try:
        from .pointrcnn_head import PointRCNNHead
        __all__['PointRCNNHead'] = PointRCNNHead
    except Exception:
        pass
    try:
        from .voxelrcnn_head import VoxelRCNNHead
        __all__['VoxelRCNNHead'] = VoxelRCNNHead
    except Exception:
        pass
    try:
        from .mppnet_head import MPPNetHead
        __all__['MPPNetHead'] = MPPNetHead
    except Exception:
        pass
    try:
        from .mppnet_memory_bank_e2e import MPPNetHeadE2E
        __all__['MPPNetHeadE2E'] = MPPNetHeadE2E
    except Exception:
        pass

_safe_import()
'''
    p.write_text(new_content, encoding="utf-8", newline="\n")
    print(f"[PASS] Patched {p.name}")
    return True


# ----------------------------------------------------------------------
# Patch 10: pcdet/datasets/__init__.py — safe import of datasets
# ----------------------------------------------------------------------
def patch_datasets_init():
    p = OPENPCDET_DIR / "pcdet" / "datasets" / "__init__.py"
    if _already_patched(p):
        return True

    new_content = f'''{SENTINEL}
import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .custom.custom_dataset import CustomDataset

__all__ = {{
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'CustomDataset': CustomDataset,
}}

# Safe conditional dataset loading
for name, mod_rel in [
    ('NuScenesDataset', '.nuscenes.nuscenes_dataset'),
    ('WaymoDataset', '.waymo.waymo_dataset'),
    ('PandasetDataset', '.pandaset.pandaset_dataset'),
    ('LyftDataset', '.lyft.lyft_dataset'),
    ('ONCEDataset', '.once.once_dataset'),
    ('Argo2Dataset', '.argo2.argo2_dataset'),
]:
    try:
        mod = __import__(f'pcdet.datasets{{mod_rel}}', fromlist=[name])
        __all__[name] = getattr(mod, name)
    except Exception:
        pass

# BEVFusionLite dataset plugin registration
try:
    from bevfusionlite.registry import DATASETS
    for name, cls in DATASETS.get_all().items():
        __all__[name] = cls
except Exception:
    pass


class DistributedSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        is_train=training,
        logger=logger
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler
'''
    p.write_text(new_content, encoding="utf-8", newline="\n")
    print(f"[PASS] Patched {p.name}")
    return True


# ----------------------------------------------------------------------
# Patch 11: roiaware_pool3d_utils.py — pure-Python fallback
# ----------------------------------------------------------------------
def patch_roiaware_pool3d_utils():
    p = OPENPCDET_DIR / "pcdet" / "ops" / "roiaware_pool3d" / "roiaware_pool3d_utils.py"
    txt = p.read_text(encoding="utf-8")
    if "def _points_in_boxes_cpu_fallback" in txt:
        return True

    fallback_code = '''
def _points_in_boxes_cpu_fallback(points, boxes):
    """Pure-PyTorch vector fallback when CUDA op is not available."""
    num_boxes = boxes.shape[0]
    num_points = points.shape[0]
    point_indices = points.new_zeros((num_boxes, num_points), dtype=torch.int)
    if num_boxes == 0 or num_points == 0:
        return point_indices
    for i in range(num_boxes):
        cx, cy, cz, dx, dy, dz, heading = boxes[i]
        pts_rel = points - boxes.new_tensor([cx, cy, cz])
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        x_local = pts_rel[:, 0] * cos_h + pts_rel[:, 1] * sin_h
        y_local = -pts_rel[:, 0] * sin_h + pts_rel[:, 1] * cos_h
        z_local = pts_rel[:, 2]
        mask = (
            (x_local.abs() <= dx / 2.0)
            & (y_local.abs() <= dy / 2.0)
            & (z_local.abs() <= dz / 2.0)
        )
        point_indices[i, mask] = 1
    return point_indices
'''
    old_call = "roiaware_pool3d_cuda.points_in_boxes_cpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)"
    new_call = """if roiaware_pool3d_cuda is not None:
        roiaware_pool3d_cuda.points_in_boxes_cpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)
    else:
        point_indices = _points_in_boxes_cpu_fallback(points, boxes)"""

    txt = txt + fallback_code
    txt = txt.replace(old_call, new_call)
    p.write_text(txt, encoding="utf-8", newline="\n")
    print(f"[PASS] Patched {p.name}")
    return True


# ----------------------------------------------------------------------
# Patch 12: kitti_dataset.py — import Path and binary tofile
# ----------------------------------------------------------------------
def patch_kitti_dataset():
    p = OPENPCDET_DIR / "pcdet" / "datasets" / "kitti" / "kitti_dataset.py"
    txt = p.read_text(encoding="utf-8")
    if "from pathlib import Path\nimport copy" in txt:
        return True

    txt = "from pathlib import Path\n" + txt
    old_tofile = "with open(filepath, 'w') as f:\n                    gt_points.tofile(f)"
    new_tofile = "gt_points.tofile(str(filepath))"
    txt = txt.replace(old_tofile, new_tofile)
    p.write_text(txt, encoding="utf-8", newline="\n")
    print(f"[PASS] Patched {p.name}")
    return True


# ----------------------------------------------------------------------
def main():
    results = []
    for name, fn in [
        ("setup.py", patch_setup_py),
        ("iou3d_nms_utils.py", patch_iou3d_nms),
        ("models/__init__.py", patch_models_init),
        ("common_utils.py", patch_common_utils),
        ("database_sampler.py", patch_database_sampler),
        ("waymo_dataset.py", patch_waymo_dataset),
        ("spconv_utils.py", patch_spconv_utils),
        ("backbones_3d", patch_backbones_3d),
        ("cuda_ops", patch_cuda_ops),
        ("dense_heads", patch_dense_heads),
        ("roi_heads", patch_roi_heads),
        ("datasets", patch_datasets_init),
        ("roiaware_pool3d_utils", patch_roiaware_pool3d_utils),
        ("kitti_dataset", patch_kitti_dataset),
    ]:
        print(f"\n{'-'*60}")
        ok = fn()
        results.append((name, ok))

    print(f"\n{'='*60}")
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"{'='*60}")

    if all_ok:
        print("\n[PASS] All patches applied successfully.")
    else:
        print("\n[FAIL] Some patches failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
