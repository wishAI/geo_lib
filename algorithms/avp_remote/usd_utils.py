def apply_gray_override(stage, root_prim_path):
    from pxr import Usd, UsdGeom, UsdShade, Sdf

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        print(f"[USD] No prim found at {root_prim_path} for gray override")
        return

    gray = (0.6, 0.6, 0.6)
    material_root_path = "/World/Materials"
    material_path = "/World/Materials/GrayOverride"
    shader_path = "/World/Materials/GrayOverride/Shader"

    UsdGeom.Scope.Define(stage, material_root_path)
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(gray)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    bound_count = 0
    for prim in Usd.PrimRange(root_prim):
        gprim = UsdGeom.Gprim(prim)
        if not gprim or not gprim.GetPrim().IsValid():
            continue
        if gprim.GetDisplayColorAttr().IsAuthored():
            gprim.GetDisplayColorAttr().Set([gray])
        else:
            gprim.CreateDisplayColorAttr([gray])
        if gprim.GetDisplayOpacityAttr().IsAuthored():
            gprim.GetDisplayOpacityAttr().Set([1.0])
        else:
            gprim.CreateDisplayOpacityAttr([1.0])
        binding_api = UsdShade.MaterialBindingAPI.Apply(gprim.GetPrim())
        try:
            binding_api.Bind(material, UsdShade.Tokens.strongerThanDescendants)
        except TypeError:
            binding_api.Bind(material)
        bound_count += 1

    print(f"[USD] Applied gray override material to {bound_count} gprims under {root_prim_path}")


def print_skeleton_info(stage, root_prim_path):
    print('start printing skeleton info')
    from pxr import UsdSkel, Usd

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        print(f"[USD] No prim found at {root_prim_path}")
        return

    print(f"[USD] Traversing skeletons under {root_prim_path}")
    found = False
    for prim in Usd.PrimRange(root_prim):
        skel = UsdSkel.Skeleton(prim)
        if not skel or not skel.GetPrim().IsValid():
            continue
        found = True
        joints = skel.GetJointsAttr().Get() or []
        rest_xforms = skel.GetRestTransformsAttr().Get() or []
        bind_xforms = skel.GetBindTransformsAttr().Get() or []
        print(f"[USD] Skeleton: {prim.GetPath()}")
        print(f"[USD]   joints: {len(joints)}")
        for name in joints:
            print(f"[USD]     {name}")
        if rest_xforms:
            print(f"[USD]   rest transforms: {len(rest_xforms)}")
        if bind_xforms:
            print(f"[USD]   bind transforms: {len(bind_xforms)}")

    if not found:
        print("[USD] No skeleton prims found")

    for prim in Usd.PrimRange(root_prim):
        anim = UsdSkel.Animation(prim)
        if not anim or not anim.GetPrim().IsValid():
            continue
        joints = anim.GetJointsAttr().Get() or []
        print(f"[USD] Animation: {prim.GetPath()}")
        if joints:
            print(f"[USD]   joints: {len(joints)}")
            for name in joints:
                print(f"[USD]     {name}")


def clear_skel_animation(stage, root_prim_path):
    from pxr import Usd, UsdSkel

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        print(f"[USD] No prim found at {root_prim_path} for animation clear")
        return

    def _clear_binding(prim):
        binding = UsdSkel.BindingAPI(prim)
        rel = binding.GetAnimationSourceRel()
        if rel and rel.HasAuthoredTargets():
            rel.ClearTargets()
            return True
        return False

    cleared = False
    for prim in Usd.PrimRange(root_prim):
        # Animation can be bound on SkelRoot or directly on Skeleton
        if UsdSkel.Root(prim):
            cleared = _clear_binding(prim) or cleared
        skel = UsdSkel.Skeleton(prim)
        if skel and skel.GetPrim().IsValid():
            cleared = _clear_binding(prim) or cleared

    if cleared:
        print("[USD] Cleared animation bindings (rest pose should be used)")
    else:
        print("[USD] No animation bindings found to clear")


def find_first_skeleton(stage, root_prim_path):
    from pxr import Usd, UsdSkel

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        return None

    for prim in Usd.PrimRange(root_prim):
        skel = UsdSkel.Skeleton(prim)
        if skel and skel.GetPrim().IsValid():
            return skel
    return None


def has_skel_animation_binding(stage, root_prim_path):
    from pxr import Usd, UsdSkel

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        return False

    def _has_binding(prim):
        binding = UsdSkel.BindingAPI(prim)
        rel = binding.GetAnimationSourceRel()
        return bool(rel and rel.HasAuthoredTargets())

    for prim in Usd.PrimRange(root_prim):
        if UsdSkel.Root(prim) and _has_binding(prim):
            return True
        skel = UsdSkel.Skeleton(prim)
        if skel and skel.GetPrim().IsValid() and _has_binding(prim):
            return True
    return False


_SKEL_BASE_REST_XFORMS = {}


def apply_rest_pose(skel, joint_rotations=None, joint_offsets=None, reset_base=False):
    """
    Apply joint rotations/offsets to a skeleton rest pose.
    Always edits against the initial rest transforms captured on first use.
    """
    from pxr import Gf

    if not skel or not skel.GetPrim().IsValid():
        print("[USD] Invalid skeleton; skipping rest pose update")
        return

    joints = skel.GetJointsAttr().Get() or []
    rest_xforms = skel.GetRestTransformsAttr().Get() or []
    if not joints or not rest_xforms:
        print("[USD] Skeleton has no joints/rest transforms to edit")
        return
    if len(joints) != len(rest_xforms):
        print("[USD] Joint/rest transform count mismatch; skipping pose edit")
        return

    skel_key = str(skel.GetPrim().GetPath())
    if reset_base or skel_key not in _SKEL_BASE_REST_XFORMS:
        _SKEL_BASE_REST_XFORMS[skel_key] = list(rest_xforms)
    base_rest_xforms = _SKEL_BASE_REST_XFORMS.get(skel_key)
    if not base_rest_xforms:
        print("[USD] No base rest transforms cached; skipping pose edit")
        return

    joint_names = [str(j) for j in joints]
    updated_joints = set()
    updated_xforms = list(base_rest_xforms)

    def _joint_index(joint_name):
        try:
            return joint_names.index(joint_name)
        except ValueError:
            print(f"[USD] Joint not found: {joint_name}")
            return None

    if joint_offsets:
        for joint_name, offset in joint_offsets.items():
            idx = _joint_index(joint_name)
            if idx is None:
                continue
            offset_mat = Gf.Matrix4d(1.0)
            offset_mat.SetTranslate(Gf.Vec3d(*offset))
            updated_xforms[idx] = updated_xforms[idx] * offset_mat
            updated_joints.add(joint_name)

    if joint_rotations:
        for joint_name, (axis, degrees) in joint_rotations.items():
            idx = _joint_index(joint_name)
            if idx is None:
                continue
            rot = Gf.Rotation(Gf.Vec3d(*axis), degrees)
            rot_mat = Gf.Matrix4d(1.0)
            rot_mat.SetRotate(rot)
            # Rest transforms are local to the parent; post-multiply to rotate in joint local space.
            updated_xforms[idx] = updated_xforms[idx] * rot_mat
            updated_joints.add(joint_name)

    if updated_joints:
        skel.GetRestTransformsAttr().Set(updated_xforms)
        print(f"[USD] Updated rest pose for {len(updated_joints)} joints")
    else:
        print("[USD] No rest pose updates applied")

def _resolve_animation_from_skeleton(skel):
    from pxr import UsdSkel

    if not skel or not skel.GetPrim().IsValid():
        return None
    binding = UsdSkel.BindingAPI(skel.GetPrim())
    rel = binding.GetAnimationSourceRel()
    if not rel or not rel.HasAuthoredTargets():
        return None
    targets = rel.GetTargets()
    if not targets:
        return None
    stage = skel.GetPrim().GetStage()
    anim_prim = stage.GetPrimAtPath(targets[0])
    anim = UsdSkel.Animation(anim_prim)
    return anim if anim and anim.GetPrim().IsValid() else None


def ensure_skel_animation_bound(skel, anim_path=None):
    """
    Ensure a UsdSkel.Animation prim exists and is bound to the skeleton.
    Returns the UsdSkel.Animation prim or None on failure.
    """
    from pxr import UsdSkel, Sdf

    if not skel or not skel.GetPrim().IsValid():
        return None

    anim = _resolve_animation_from_skeleton(skel)
    if anim and anim.GetPrim().IsValid():
        return anim

    stage = skel.GetPrim().GetStage()
    if not stage:
        return None

    if anim_path is None:
        anim_path = skel.GetPrim().GetPath().AppendChild("SkelAnimation")
    else:
        anim_path = Sdf.Path(anim_path)

    anim = UsdSkel.Animation.Define(stage, anim_path)
    if not anim or not anim.GetPrim().IsValid():
        return None

    joints = skel.GetJointsAttr().Get() or []
    if joints:
        anim.GetJointsAttr().Set(joints)

    binding = UsdSkel.BindingAPI.Apply(skel.GetPrim())
    binding.CreateAnimationSourceRel().SetTargets([anim.GetPrim().GetPath()])
    return anim


def apply_animation_pose(skel, joint_rotations, time_code):
    """Apply joint rotations to the animation bound to a skeleton."""
    from pxr import Gf, UsdSkel

    if not skel or not skel.GetPrim().IsValid():
        print("[USD] Invalid skeleton; skipping animation pose update")
        return

    skel_anim = ensure_skel_animation_bound(skel)
    if not skel_anim or not skel_anim.GetPrim().IsValid():
        print("[USD] No UsdSkel.Animation bound; skipping animation pose update")
        return

    joints = skel_anim.GetJointsAttr().Get() or []
    if not joints:
        skel_joints = skel.GetJointsAttr().Get() or []
        if skel_joints:
            skel_anim.GetJointsAttr().Set(skel_joints)
            joints = skel_joints
    if not joints:
        print("[USD] Animation has no joints; skipping animation pose update")
        return

    current_rotations = skel_anim.GetRotationsAttr().Get(time_code)
    if not current_rotations or len(current_rotations) != len(joints):
        current_rotations = [Gf.Quatf(1, 0, 0, 0)] * len(joints)

    joint_names = [str(j) for j in joints]
    changed = 0
    for joint_name, (axis, degrees) in joint_rotations.items():
        if joint_name not in joint_names:
            continue
        idx = joint_names.index(joint_name)
        rot = Gf.Rotation(Gf.Vec3d(*axis), degrees)
        new_quat = Gf.Quatf(rot.GetQuat())
        if current_rotations[idx] != new_quat:
            changed += 1
        current_rotations[idx] = new_quat

    skel_anim.GetRotationsAttr().Set(current_rotations, time_code)
    print(f"[USD] Animation pose updated ({changed} joints) at time {time_code}")


def mat4_to_euler_xyz(mat4):
    import math
    import numpy as np

    r = mat4[:3, :3]
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


def mat4_rotation_angle_deg(mat4):
    import math
    import numpy as np

    r = mat4[:3, :3]
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    cos_angle = (trace - 1.0) * 0.5
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))
