from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R


THIS_DIR = Path(__file__).resolve().parent
VIDEO_TO_SONIC_DIR = THIS_DIR.parent
FOURD_ROOT = VIDEO_TO_SONIC_DIR / "4D-Humans"

if str(FOURD_ROOT) not in sys.path:
    sys.path.insert(0, str(FOURD_ROOT))

from hmr2.configs import CACHE_DIR_4DHUMANS  # noqa: E402
from hmr2.datasets.vitdet_dataset import ViTDetDataset  # noqa: E402
from hmr2.models import DEFAULT_CHECKPOINT, check_smpl_exists, download_models, load_hmr2  # noqa: E402
from hmr2.utils import recursive_to  # noqa: E402
from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a primary-subject SMPL sequence from a folder of frames."
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--img_folder", type=str, required=True)
    parser.add_argument("--out_npz", type=str, required=True)
    parser.add_argument("--detector", type=str, default="regnety", choices=["vitdet", "regnety"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--file_type", nargs="+", default=["*.jpg", "*.png"])
    parser.add_argument(
        "--min_box_area_ratio",
        type=float,
        default=0.002,
        help="Reject person boxes smaller than this fraction of the full image area.",
    )
    parser.add_argument(
        "--min_track_area_ratio",
        type=float,
        default=0.2,
        help="Reject tracked boxes that collapse below this fraction of the previous subject area.",
    )
    parser.add_argument(
        "--max_center_jump_ratio",
        type=float,
        default=0.25,
        help="Reject track candidates that jump farther than this fraction of the image diagonal.",
    )
    parser.add_argument(
        "--track_reset_after",
        type=int,
        default=3,
        help="Reset the temporal tracker after this many consecutive missed frames.",
    )
    return parser.parse_args()


def _pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no CUDA device is available")
    return torch.device(requested)


def _build_detector(detector_name: str, device: torch.device):
    if detector_name == "vitdet":
        from detectron2.config import LazyConfig
        import hmr2

        cfg_path = Path(hmr2.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.runtime_device = str(device)
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        return DefaultPredictor_Lazy(detectron2_cfg)

    from detectron2 import model_zoo

    detectron2_cfg = model_zoo.get_config(
        "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
    )
    detectron2_cfg.runtime_device = str(device)
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
    return DefaultPredictor_Lazy(detectron2_cfg)


def _rotmats_to_axis_angle(rotmats: np.ndarray) -> np.ndarray:
    rotvec = R.from_matrix(rotmats.reshape(-1, 3, 3)).as_rotvec()
    return rotvec.reshape(*rotmats.shape[:-2], 3).astype(np.float32)


def _load_raw_smpl(model_cfg, device: torch.device) -> smplx.SMPLLayer:
    model_path = Path(CACHE_DIR_4DHUMANS) / "data" / "smpl"
    smpl = smplx.SMPLLayer(
        model_path=str(model_path),
        gender=str(model_cfg.SMPL.GENDER),
        num_body_joints=int(model_cfg.SMPL.NUM_BODY_JOINTS),
    )
    return smpl.to(device).eval()


def _get_frame_paths(img_folder: Path, file_types: list[str], max_frames: int) -> list[Path]:
    frame_paths: set[Path] = set()
    for pattern in file_types:
        frame_paths.update(img_folder.glob(pattern))
    ordered = sorted(frame_paths)
    if max_frames > 0:
        ordered = ordered[:max_frames]
    return ordered


def _box_areas(boxes: np.ndarray) -> np.ndarray:
    widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0.0, a_max=None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0.0, a_max=None)
    return widths * heights


def _box_centers(boxes: np.ndarray) -> np.ndarray:
    return np.stack(((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5), axis=1)


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
    inter_area = inter_w * inter_h

    area_a = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    area_b = _box_areas(boxes)
    union = np.clip(area_a + area_b - inter_area, a_min=1e-6, a_max=None)
    return inter_area / union


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _print_progress(
    processed: int,
    total: int,
    kept: int,
    skipped: int,
    start_time: float,
    *,
    done: bool = False,
) -> None:
    elapsed = max(time.monotonic() - start_time, 1e-6)
    rate = processed / elapsed
    remaining = max(total - processed, 0)
    eta = remaining / max(rate, 1e-6)
    fraction = processed / max(total, 1)
    bar_width = 28
    filled = min(bar_width, int(round(fraction * bar_width)))
    bar = "#" * filled + "-" * (bar_width - filled)
    message = (
        f"[{bar}] {processed}/{total} {fraction * 100:5.1f}% "
        f"kept={kept} skipped={skipped} "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}"
    )

    if sys.stderr.isatty():
        print(f"\r{message}", end="\n" if done else "", file=sys.stderr, flush=True)
    else:
        if done or processed == total or processed == 1 or processed % 10 == 0:
            print(message, file=sys.stderr, flush=True)


def _select_primary_box(
    boxes: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
    min_box_area_ratio: float,
    min_track_area_ratio: float,
    max_center_jump_ratio: float,
    prev_box: np.ndarray | None,
    prev_center: np.ndarray | None,
    prev_velocity: np.ndarray,
    missed_frames: int,
) -> tuple[int | None, dict[str, float | int]]:
    image_h, image_w = image_shape
    image_area = float(image_h * image_w)
    image_diag = float(np.hypot(image_w, image_h))
    areas = _box_areas(boxes)
    centers = _box_centers(boxes)
    min_abs_area = image_area * min_box_area_ratio

    info: dict[str, float | int] = {
        "mode": 0,
        "score": 0.0,
        "iou": 0.0,
        "center_jump_px": 0.0,
    }

    if prev_box is None or prev_center is None:
        candidate_mask = areas >= min_abs_area
        if not np.any(candidate_mask):
            return None, info
        candidate_idx = np.flatnonzero(candidate_mask)
        best_local = int(np.argmax(areas[candidate_mask] * scores[candidate_mask]))
        best_idx = int(candidate_idx[best_local])
        info["mode"] = 0
        return best_idx, info

    prev_area = max((prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1]), 1.0)
    predicted_center = prev_center + prev_velocity * max(1, missed_frames + 1)
    center_dists = np.linalg.norm(centers - predicted_center[None, :], axis=1)
    center_dists_norm = center_dists / max(image_diag, 1.0)
    area_ratio = np.maximum(areas, 1.0) / prev_area
    area_penalty = np.abs(np.log(area_ratio))
    ious = _box_iou(prev_box, boxes)

    candidate_mask = (areas >= np.maximum(min_abs_area, prev_area * min_track_area_ratio)) & (
        center_dists_norm <= max_center_jump_ratio
    )
    if not np.any(candidate_mask):
        candidate_mask = (
            (areas >= min_abs_area)
            & ((ious >= 0.05) | (center_dists_norm <= max_center_jump_ratio * 0.75))
        )
    if not np.any(candidate_mask):
        return None, info

    track_scores = (
        4.0 * ious
        - 3.0 * center_dists_norm
        - 0.75 * area_penalty
        + 0.5 * scores
        + 0.15 * np.clip(area_ratio, a_min=0.0, a_max=2.0)
    )

    candidate_idx = np.flatnonzero(candidate_mask)
    best_local = int(np.argmax(track_scores[candidate_mask]))
    best_idx = int(candidate_idx[best_local])

    info["mode"] = 1
    info["score"] = float(track_scores[best_idx])
    info["iou"] = float(ious[best_idx])
    info["center_jump_px"] = float(center_dists[best_idx])
    return best_idx, info


def main() -> None:
    args = _parse_args()
    device = _pick_device(args.device)

    img_folder = Path(args.img_folder).resolve()
    out_npz = Path(args.out_npz).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = _get_frame_paths(img_folder, args.file_type, args.max_frames)
    if not frame_paths:
        raise FileNotFoundError(f"No input frames found in {img_folder}")

    download_models(CACHE_DIR_4DHUMANS)
    check_smpl_exists()
    model, model_cfg = load_hmr2(args.checkpoint)
    model = model.to(device).eval()
    detector = _build_detector(args.detector, device)
    raw_smpl = _load_raw_smpl(model_cfg, device)

    export = {
        "frame_name": [],
        "frame_path": [],
        "frame_index": [],
        "bbox_xyxy": [],
        "score": [],
        "pred_cam": [],
        "pred_cam_t": [],
        "betas": [],
        "global_orient_axis_angle": [],
        "body_pose_23_axis_angle": [],
        "smpl_pose_21_axis_angle": [],
        "smpl_joints_world": [],
        "smpl_joints_local": [],
        "track_mode": [],
        "track_score": [],
        "track_iou": [],
        "track_center_jump_px": [],
    }

    skipped = 0
    total = len(frame_paths)
    track_misses = 0
    prev_box: np.ndarray | None = None
    prev_center: np.ndarray | None = None
    prev_velocity = np.zeros(2, dtype=np.float32)
    tiny_or_inconsistent_rejects = 0
    tracker_resets = 0
    start_time = time.monotonic()

    for frame_index, img_path in enumerate(frame_paths):
        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            skipped += 1
            _print_progress(
                frame_index + 1, total, len(export["frame_name"]), skipped, start_time
            )
            continue

        det_out = detector(img_cv2)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > args.score_thresh)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        scores = det_instances.scores[valid_idx].cpu().numpy()

        if len(boxes) == 0:
            skipped += 1
            _print_progress(
                frame_index + 1, total, len(export["frame_name"]), skipped, start_time
            )
            continue

        best_idx, track_info = _select_primary_box(
            boxes=boxes,
            scores=scores,
            image_shape=img_cv2.shape[:2],
            min_box_area_ratio=args.min_box_area_ratio,
            min_track_area_ratio=args.min_track_area_ratio,
            max_center_jump_ratio=args.max_center_jump_ratio,
            prev_box=prev_box,
            prev_center=prev_center,
            prev_velocity=prev_velocity,
            missed_frames=track_misses,
        )
        if best_idx is None:
            skipped += 1
            track_misses += 1
            tiny_or_inconsistent_rejects += 1
            if track_misses >= args.track_reset_after:
                prev_box = None
                prev_center = None
                prev_velocity = np.zeros(2, dtype=np.float32)
                tracker_resets += 1
            _print_progress(
                frame_index + 1, total, len(export["frame_name"]), skipped, start_time
            )
            continue

        best_box = boxes[best_idx : best_idx + 1]
        best_score = float(scores[best_idx])
        current_center = _box_centers(best_box)[0]

        dataset = ViTDetDataset(model_cfg, img_cv2, best_box)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(1, args.batch_size),
            shuffle=False,
            num_workers=0,
        )

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

                pred_smpl_params = out["pred_smpl_params"]
                body_pose_rotmat = pred_smpl_params["body_pose"].detach()
                global_orient_rotmat = pred_smpl_params["global_orient"].detach()
                betas = pred_smpl_params["betas"].detach()

                smpl_output = raw_smpl(
                    global_orient=global_orient_rotmat.float(),
                    body_pose=body_pose_rotmat.float(),
                    betas=betas.float(),
                    pose2rot=False,
                )
                joints_world = smpl_output.joints[:, :24, :].detach()
                joints_centered = joints_world - joints_world[:, :1, :]
                root_inv = global_orient_rotmat[:, 0].transpose(1, 2)
                joints_local = torch.einsum("bij,bkj->bki", root_inv, joints_centered)

            global_orient_aa = _rotmats_to_axis_angle(
                global_orient_rotmat.cpu().numpy().astype(np.float64)
            )[:, 0]
            body_pose_aa = _rotmats_to_axis_angle(
                body_pose_rotmat.cpu().numpy().astype(np.float64)
            )

            export["frame_name"].append(img_path.name)
            export["frame_path"].append(str(img_path))
            export["frame_index"].append(frame_index)
            export["bbox_xyxy"].append(best_box[0].astype(np.float32))
            export["score"].append(best_score)
            export["pred_cam"].append(out["pred_cam"].detach().cpu().numpy()[0].astype(np.float32))
            export["pred_cam_t"].append(out["pred_cam_t"].detach().cpu().numpy()[0].astype(np.float32))
            export["betas"].append(betas.cpu().numpy()[0].astype(np.float32))
            export["global_orient_axis_angle"].append(global_orient_aa.astype(np.float32))
            export["body_pose_23_axis_angle"].append(body_pose_aa[0].astype(np.float32))
            export["smpl_pose_21_axis_angle"].append(body_pose_aa[0, :21].astype(np.float32))
            export["smpl_joints_world"].append(joints_world.cpu().numpy()[0].astype(np.float32))
            export["smpl_joints_local"].append(joints_local.cpu().numpy()[0].astype(np.float32))
            export["track_mode"].append(int(track_info["mode"]))
            export["track_score"].append(np.float32(track_info["score"]))
            export["track_iou"].append(np.float32(track_info["iou"]))
            export["track_center_jump_px"].append(np.float32(track_info["center_jump_px"]))

        if prev_center is not None:
            observed_velocity = current_center - prev_center
            prev_velocity = 0.6 * prev_velocity + 0.4 * observed_velocity.astype(np.float32)
        prev_box = best_box[0].astype(np.float32)
        prev_center = current_center.astype(np.float32)
        track_misses = 0
        _print_progress(frame_index + 1, total, len(export["frame_name"]), skipped, start_time)

    kept = len(export["frame_name"])
    if kept == 0:
        raise RuntimeError("No valid person detections were exported.")

    _print_progress(total, total, kept, skipped, start_time, done=True)

    np.savez_compressed(
        out_npz,
        frame_name=np.asarray(export["frame_name"], dtype=str),
        frame_path=np.asarray(export["frame_path"], dtype=str),
        frame_index=np.asarray(export["frame_index"], dtype=np.int32),
        bbox_xyxy=np.asarray(export["bbox_xyxy"], dtype=np.float32),
        score=np.asarray(export["score"], dtype=np.float32),
        pred_cam=np.asarray(export["pred_cam"], dtype=np.float32),
        pred_cam_t=np.asarray(export["pred_cam_t"], dtype=np.float32),
        betas=np.asarray(export["betas"], dtype=np.float32),
        global_orient_axis_angle=np.asarray(export["global_orient_axis_angle"], dtype=np.float32),
        body_pose_23_axis_angle=np.asarray(export["body_pose_23_axis_angle"], dtype=np.float32),
        smpl_pose_21_axis_angle=np.asarray(export["smpl_pose_21_axis_angle"], dtype=np.float32),
        smpl_joints_world=np.asarray(export["smpl_joints_world"], dtype=np.float32),
        smpl_joints_local=np.asarray(export["smpl_joints_local"], dtype=np.float32),
        track_mode=np.asarray(export["track_mode"], dtype=np.int32),
        track_score=np.asarray(export["track_score"], dtype=np.float32),
        track_iou=np.asarray(export["track_iou"], dtype=np.float32),
        track_center_jump_px=np.asarray(export["track_center_jump_px"], dtype=np.float32),
    )

    print(f"Exported {kept} frames to {out_npz}")
    print(f"Skipped {skipped} / {total} frames with no valid primary subject")
    print(
        f"Tracker resets: {tracker_resets}, "
        f"rejected tiny/inconsistent detections: {tiny_or_inconsistent_rejects}"
    )
    print(
        "Saved keys: smpl_pose_21_axis_angle, smpl_joints_local, "
        "joint camera data, tracker diagnostics, and metadata"
    )


if __name__ == "__main__":
    main()
