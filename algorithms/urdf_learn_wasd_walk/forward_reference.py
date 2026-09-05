"""Pure command-gated phase reference for the Landau v3 gait probe."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from xml.etree import ElementTree as ET

from algorithms.urdf_learn_wasd_walk import forward_walk_contract as contract
from algorithms.urdf_learn_wasd_walk import model_spec


METHOD_ID = "command_gated_phase_reference_residual_v3"


@dataclass(frozen=True)
class ReferenceConfig:
    period_s: float = contract.GAIT_PERIOD_S
    phase_difference_cycles: float = 0.5
    amplitude_scale: float = 1.0
    startup_ramp_s: float = 0.4
    swing_fraction: float = 0.5
    hip_pitch_amplitude: float = 0.2
    knee_amplitude: float = 1.0
    ankle_pitch_amplitude: float = -1.0
    toe_amplitude: float = -0.5
    hip_roll_amplitude: float = 0.0
    hip_phase_offset_cycles: float = 0.0
    knee_phase_offset_cycles: float = 0.0
    ankle_phase_offset_cycles: float = 0.0
    toe_phase_offset_cycles: float = 0.0

    def validate(self) -> None:
        if self.period_s <= 0.0:
            raise ValueError("reference period must be positive")
        if not 0.0 < self.swing_fraction < 1.0:
            raise ValueError("swing fraction must be between zero and one")
        if self.startup_ramp_s < 0.0:
            raise ValueError("startup ramp cannot be negative")
        if not 0.0 <= self.amplitude_scale <= 1.0:
            raise ValueError("amplitude scale must stay in [0, 1]")
        for name in (
            "hip_pitch_amplitude", "knee_amplitude", "ankle_pitch_amplitude",
            "toe_amplitude",
            "hip_roll_amplitude",
        ):
            if abs(float(getattr(self, name))) > 1.0:
                raise ValueError(f"{name} escapes the normalized action limit")

    def as_metadata(self) -> dict:
        self.validate()
        return asdict(self)


def _smoothstep01(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def command_gate(forward_command_mps: float) -> float:
    """Return exactly zero at stand and saturate at the v2 training command floor."""

    return _smoothstep01(abs(float(forward_command_mps)) / contract.TRAIN_FORWARD_SPEED_RANGE_MPS[0])


def _swing_clearance(phase: float, swing_fraction: float) -> float:
    phase %= 1.0
    if phase >= swing_fraction:
        return 0.0
    # sin² has zero value and zero slope at toe-off and landing.  That keeps
    # the open-loop probe from commanding an impact discontinuity.
    return math.sin(math.pi * phase / swing_fraction) ** 2


def reference_action(
    time_s: float,
    forward_command_mps: float,
    config: ReferenceConfig = ReferenceConfig(),
) -> list[float]:
    """Return normalized open-loop actions in the canonical action-joint order."""

    config.validate()
    if time_s < 0.0:
        raise ValueError("reference time cannot be negative")
    gate = command_gate(forward_command_mps)
    ramp = 1.0 if config.startup_ramp_s == 0.0 else _smoothstep01(time_s / config.startup_ramp_s)
    scale = gate * ramp * config.amplitude_scale
    result = {name: 0.0 for name in model_spec.ACTION_JOINTS}
    direction = 1.0 if forward_command_mps >= 0.0 else -1.0
    base_phase = time_s / config.period_s
    for side, side_offset in (("left", 0.0), ("right", config.phase_difference_cycles)):
        phase = (base_phase + side_offset) % 1.0
        hip_phase = (phase + config.hip_phase_offset_cycles) % 1.0
        knee_phase = (phase + config.knee_phase_offset_cycles) % 1.0
        ankle_phase = (phase + config.ankle_phase_offset_cycles) % 1.0
        toe_phase = (phase + config.toe_phase_offset_cycles) % 1.0
        result[f"{side}_hip_pitch_joint"] = (
            scale * direction * config.hip_pitch_amplitude * math.sin(2.0 * math.pi * hip_phase)
        )
        result[f"{side}_hip_roll_joint"] = (
            scale * config.hip_roll_amplitude * math.sin(2.0 * math.pi * phase)
        )
        result[f"{side}_knee_joint"] = (
            scale * config.knee_amplitude * _swing_clearance(knee_phase, config.swing_fraction)
        )
        result[f"{side}_ankle_pitch_joint"] = (
            scale * config.ankle_pitch_amplitude * _swing_clearance(ankle_phase, config.swing_fraction)
        )
        result[f"{side}_toe_joint"] = (
            scale * config.toe_amplitude * _swing_clearance(toe_phase, config.swing_fraction)
        )
    actions = [result[name] for name in model_spec.ACTION_JOINTS]
    if max(map(abs, actions), default=0.0) > contract.ACTION_CLIP + 1.0e-12:
        raise ValueError("reference action escaped the canonical action clip")
    return actions


def reference_contract(config: ReferenceConfig) -> dict:
    """Machine-readable semantics for the first bounded v3 experiment."""

    return {
        "method": METHOD_ID,
        "policy_residual_enabled": False,
        "camera_enabled": False,
        "semantic_forward_axis": "+Y",
        "command_gating": "smoothstep(abs(forward_mps) / 0.25); exactly zero at zero command",
        "left_right_phase_difference_cycles": config.phase_difference_cycles,
        "swing_clearance_envelope": "sin(pi*u)^2 during swing; zero value and slope at toe-off/landing",
        "lateral_weight_transfer": (
            "phase-opposed bilateral hip-roll sinus; amplitude is zero in the baseline and the "
            "entire reference remains exactly zero at zero command"
        ),
        "action_scale_rad": contract.ACTION_SCALE_RAD,
        "action_order": list(model_spec.ACTION_JOINTS),
        "parameters": config.as_metadata(),
        "offline_kinematic_audit": offline_kinematic_audit(config),
        "acceptance": {
            "bilateral_direct_air_run_min_control_steps": 2,
            "bilateral_support_body_height_gain_min_m": 0.002,
            "semantic_forward_displacement_min_m": 0.0,
            "reset_count": 0,
            "done_count": 0,
            "fall_count": 0,
            "max_joint_target_error_rad": 0.35,
        },
    }


def offline_kinematic_audit(config: ReferenceConfig) -> dict:
    """Measure theoretical swing clearance with exact URDF FK/collision meshes."""

    root = ET.parse(model_spec.URDF_PATH).getroot()
    nominal = dict(model_spec.CANONICAL_NOMINAL_NONFINGER_POSITIONS_RAD)
    nominal.update({name: 0.0 for name in model_spec.FINGER_JOINTS})
    transforms, _ = model_spec._joint_world_transforms(root, nominal)
    nominal_points = model_spec._collision_world_points(root, model_spec.URDF_PATH, transforms)
    full_config = replace(config, startup_ramp_s=0.0)
    per_side = {}
    for side, time_s, suffix in (
        ("left", config.period_s * 0.25, "l"),
        ("right", config.period_s * 0.75, "r"),
    ):
        action = reference_action(time_s, contract.TARGET_FORWARD_SPEED_MPS, full_config)
        posed = dict(nominal)
        for name, normalized in zip(model_spec.ACTION_JOINTS, action):
            posed[name] = posed.get(name, 0.0) + normalized * contract.ACTION_SCALE_RAD
        posed_transforms, _ = model_spec._joint_world_transforms(root, posed)
        posed_points = model_spec._collision_world_points(
            root, model_spec.URDF_PATH, posed_transforms
        )
        names = (f"foot_{suffix}", f"toes_01_{suffix}")
        nominal_side = [point for name in names for point in nominal_points[name]]
        posed_side = [point for name in names for point in posed_points[name]]
        per_side[side] = {
            "sample_phase_cycles": 0.25,
            "minimum_collision_height_gain_m": round(
                min(point[2] for point in posed_side) - min(point[2] for point in nominal_side), 9
            ),
            "mean_collision_forward_shift_m": round(
                sum(point[1] for point in posed_side) / len(posed_side)
                - sum(point[1] for point in nominal_side) / len(nominal_side),
                9,
            ),
        }
    return {
        "method": "exact current-URDF FK and collision-mesh vertices",
        "urdf_sha256": model_spec.EXPECTED_URDF_SHA256,
        "action_scale_rad": contract.ACTION_SCALE_RAD,
        "full_amplitude_mid_swing": per_side,
        "note": "runtime contact and body-height observations remain authoritative",
    }


def evaluate_probe(metrics: dict, reference: dict) -> list[str]:
    acceptance = reference["acceptance"]
    failures: list[str] = []
    for side in ("left", "right"):
        if int(metrics.get(f"{side}_max_consecutive_direct_air_steps", 0)) < int(
            acceptance["bilateral_direct_air_run_min_control_steps"]
        ):
            failures.append(f"{side} side had no robust direct-observation air interval")
        if float(metrics.get(f"{side}_max_support_body_height_gain_m", -math.inf)) < float(
            acceptance["bilateral_support_body_height_gain_min_m"]
        ):
            failures.append(f"{side} support geometry did not reach the clearance threshold")
    if float(metrics.get("semantic_forward_displacement_m", -math.inf)) <= float(
        acceptance["semantic_forward_displacement_min_m"]
    ):
        failures.append("open-loop reference produced no positive semantic +Y displacement")
    for name in ("reset_count", "done_count", "fall_count"):
        if int(metrics.get(name, -1)) != int(acceptance[name]):
            failures.append(f"{name}={metrics.get(name)}")
    if float(metrics.get("max_joint_target_error_rad", math.inf)) > float(
        acceptance["max_joint_target_error_rad"]
    ):
        failures.append("joint target error exceeded the bounded-reference criterion")
    return failures
