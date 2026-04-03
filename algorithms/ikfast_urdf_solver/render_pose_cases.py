from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .benchmark import MujocoChain, sample_joint_values
from .config import ALGORITHM_ROOT, get_builtin_config, list_builtin_configs, load_config_from_json
from .solver import IkFastSolver, pose_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MuJoCo screenshots comparing IKFast and pytracik on the same target poses.")
    parser.add_argument("--config", default="ur5_helper_sample", help=f"Built-in config name or path to a JSON config. Built-ins: {', '.join(list_builtin_configs())}")
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seed-mode", choices=("truth", "zero"), default="zero")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--output-dir", type=Path, default=ALGORITHM_ROOT / "outputs" / "pose_renders")
    args = parser.parse_args()

    config = get_builtin_config(args.config) if args.config in list_builtin_configs() else load_config_from_json(args.config)
    summary = render_pose_cases(
        config,
        cases=args.cases,
        seed=args.seed,
        seed_mode=args.seed_mode,
        width=args.width,
        height=args.height,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


def render_pose_cases(
    config,
    *,
    cases: int,
    seed: int,
    seed_mode: str,
    width: int,
    height: int,
    output_dir: Path,
) -> dict[str, object]:
    from trac_ik import TracIK

    solver = IkFastSolver(config)
    mujoco_chain = MujocoChain.from_solver(solver)
    pytracik_solver = TracIK(
        base_link_name=solver.config.base_link,
        tip_link_name=solver.config.tip_link,
        urdf_path=str(mujoco_chain.sanitized_urdf_path),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    summary_cases: list[dict[str, object]] = []
    generated_files: list[str] = []

    for case_index in range(cases):
        target_joint_values = sample_joint_values(rng, solver.lower_bounds, solver.upper_bounds, solver.continuous_mask)
        target_position, target_rotation = mujoco_chain.forward_kinematics(target_joint_values)
        seed_values = target_joint_values if seed_mode == "truth" else np.zeros_like(target_joint_values)

        ikfast_result = solver.ik(target_position, target_rotation, seed_joint_values=seed_values)
        pytracik_joint_values = pytracik_solver.ik(target_position, target_rotation, seed_jnt_values=seed_values)
        if not ikfast_result.success or ikfast_result.joint_values is None:
            raise RuntimeError(f"IKFast failed to solve case {case_index + 1}.")
        if pytracik_joint_values is None:
            raise RuntimeError(f"pytracik failed to solve case {case_index + 1}.")

        ikfast_joint_values = np.asarray(ikfast_result.joint_values, dtype=np.float64)
        pytracik_joint_values = np.asarray(pytracik_joint_values, dtype=np.float64)

        ikfast_points = mujoco_chain.chain_points(ikfast_joint_values)
        pytracik_points = mujoco_chain.chain_points(pytracik_joint_values)
        target_tip_position = np.asarray(target_position, dtype=np.float64)
        ikfast_tip_error = pose_error(target_position, target_rotation, *mujoco_chain.forward_kinematics(ikfast_joint_values))
        pytracik_tip_error = pose_error(target_position, target_rotation, *mujoco_chain.forward_kinematics(pytracik_joint_values))

        scene_xml = build_static_comparison_scene(
            ikfast_points=ikfast_points,
            pytracik_points=pytracik_points,
            target_tip_position=target_tip_position,
            title=f"case_{case_index + 1:02d}",
        )
        model = mujoco.MjModel.from_xml_string(scene_xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        image = render_scene_image(model, data, width=width, height=height)
        annotated = annotate_image(
            image,
            [
                f"Case {case_index + 1:02d}",
                "Blue: IKFast",
                "Orange: pytracik",
                f"IKFast pos/rot err: {ikfast_tip_error[0]:.2e} m / {ikfast_tip_error[1]:.2e} rad",
                f"pytracik pos/rot err: {pytracik_tip_error[0]:.2e} m / {pytracik_tip_error[1]:.2e} rad",
            ],
        )
        image_path = output_dir / f"case_{case_index + 1:02d}.png"
        annotated.save(image_path)
        generated_files.append(str(image_path))
        summary_cases.append(
            {
                "case_index": case_index + 1,
                "image_path": str(image_path),
                "target_joint_values": target_joint_values.tolist(),
                "seed_joint_values": seed_values.tolist(),
                "ikfast_joint_values": ikfast_joint_values.tolist(),
                "pytracik_joint_values": pytracik_joint_values.tolist(),
                "ikfast_position_error": ikfast_tip_error[0],
                "ikfast_rotation_error": ikfast_tip_error[1],
                "pytracik_position_error": pytracik_tip_error[0],
                "pytracik_rotation_error": pytracik_tip_error[1],
            }
        )

    summary = {
        "config": {
            "name": solver.config.name,
            "urdf_path": str(solver.config.urdf_path),
            "base_link": solver.config.base_link,
            "tip_link": solver.config.tip_link,
        },
        "seed": seed,
        "seed_mode": seed_mode,
        "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        "cases": summary_cases,
        "generated_files": generated_files,
    }
    summary_path = output_dir / "pose_cases_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    generated_files.append(str(summary_path))

    contact_sheet_path = output_dir / "contact_sheet.png"
    build_contact_sheet([Path(case["image_path"]) for case in summary_cases], contact_sheet_path)
    generated_files.append(str(contact_sheet_path))
    summary["generated_files"] = generated_files
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def build_static_comparison_scene(
    *,
    ikfast_points: np.ndarray,
    pytracik_points: np.ndarray,
    target_tip_position: np.ndarray,
    title: str,
) -> str:
    left_offset = np.array([0.0, -0.7, 0.0], dtype=np.float64)
    right_offset = np.array([0.0, 0.7, 0.0], dtype=np.float64)
    lines = [
        "<mujoco model='ikfast_pose_compare'>",
        "  <option timestep='0.01' gravity='0 0 -9.81'/>",
        "  <visual>",
        "    <headlight ambient='0.45 0.45 0.45' diffuse='0.85 0.85 0.85' specular='0.15 0.15 0.15'/>",
        "    <global offwidth='1440' offheight='960'/>",
        "    <rgba haze='0.15 0.2 0.25 1'/>",
        "  </visual>",
        "  <asset>",
        "    <texture name='grid' type='2d' builtin='checker' rgb1='0.18 0.2 0.24' rgb2='0.12 0.14 0.18' width='512' height='512'/>",
        "    <material name='floor' texture='grid' texrepeat='8 8' reflectance='0.1' rgba='0.9 0.9 0.9 1'/>",
        "  </asset>",
        "  <worldbody>",
        "    <geom name='floor' type='plane' size='3 3 0.1' pos='0 0 -0.02' material='floor'/>",
        "    <light name='key' pos='2.5 -2.0 3.0' dir='-0.4 0.3 -1' diffuse='1 1 1' specular='0.25 0.25 0.25'/>",
        "    <light name='fill' pos='-2.0 2.0 2.5' dir='0.3 -0.2 -1' diffuse='0.55 0.55 0.6' specular='0.1 0.1 0.1'/>",
    ]
    lines.extend(_capsule_geoms("ikfast", ikfast_points + left_offset, rgba=(0.18, 0.49, 0.90, 1.0)))
    lines.extend(_capsule_geoms("pytracik", pytracik_points + right_offset, rgba=(0.95, 0.47, 0.18, 1.0)))
    lines.extend(_marker_geoms("ikfast_target", target_tip_position + left_offset, rgba=(0.2, 0.95, 0.35, 1.0)))
    lines.extend(_marker_geoms("pytracik_target", target_tip_position + right_offset, rgba=(0.2, 0.95, 0.35, 1.0)))
    lines.append("  </worldbody>")
    lines.append("</mujoco>")
    return "\n".join(lines)


def _capsule_geoms(prefix: str, points: np.ndarray, *, rgba: tuple[float, float, float, float]) -> list[str]:
    lines: list[str] = []
    rgba_text = " ".join(f"{value:.3f}" for value in rgba)
    for index, point in enumerate(points):
        px, py, pz = point
        lines.append(
            f"    <geom name='{prefix}_joint_{index:02d}' type='sphere' size='0.026' pos='{px:.6f} {py:.6f} {pz:.6f}' rgba='{rgba_text}'/>"
        )
    for index in range(len(points) - 1):
        p0 = points[index]
        p1 = points[index + 1]
        if np.linalg.norm(p1 - p0) < 1e-9:
            continue
        fromto = " ".join(f"{value:.6f}" for value in np.concatenate([p0, p1]))
        lines.append(
            f"    <geom name='{prefix}_link_{index:02d}' type='capsule' size='0.018' fromto='{fromto}' rgba='{rgba_text}'/>"
        )
    return lines


def _marker_geoms(prefix: str, point: np.ndarray, *, rgba: tuple[float, float, float, float]) -> list[str]:
    rgba_text = " ".join(f"{value:.3f}" for value in rgba)
    px, py, pz = point
    return [
        f"    <geom name='{prefix}' type='sphere' size='0.034' pos='{px:.6f} {py:.6f} {pz:.6f}' rgba='{rgba_text}'/>"
    ]


def render_scene_image(model: mujoco.MjModel, data: mujoco.MjData, *, width: int, height: int) -> Image.Image:
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    camera.lookat = np.array([0.0, 0.0, 0.55], dtype=np.float64)
    camera.distance = 2.7
    camera.azimuth = 132.0
    camera.elevation = -24.0
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    renderer.close()
    return Image.fromarray(pixels)


def annotate_image(image: Image.Image, lines: list[str]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    line_height = 28
    margin = 24
    box_height = margin * 2 + (line_height * len(lines))
    draw.rounded_rectangle(
        (18, 18, 730, 18 + box_height),
        radius=18,
        fill=(8, 12, 18, 170),
    )
    y = 18 + margin
    for line in lines:
        draw.text((36, y), line, fill=(245, 248, 252, 255))
        y += line_height
    return annotated


def build_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        raise ValueError("No images provided for contact sheet.")
    columns = 2
    rows = (len(images) + columns - 1) // columns
    cell_width, cell_height = images[0].size
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), color=(18, 20, 24))
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        sheet.paste(image, (col * cell_width, row * cell_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


if __name__ == "__main__":
    main()
