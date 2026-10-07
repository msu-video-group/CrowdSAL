"""Evaluate predicted saliency videos against CrowdSAL annotations.

Input includes model predictions and GT saliency maps in MP4 format and
per-video fixations in JSON. Fixation files contain one non-empty list of
zero-based ``[y, x]`` coordinates per video frame. Repeated coordinates are
preserved as separate observer events.

The benchmark calculates CC, SIM, NSS, and event-weighted AUC-Judd for every
frame. Prediction and ground-truth dimensions and frame counts must match; the
benchmark does not resize inputs. Results are written per video, together with
frame-weighted and video-weighted dataset means in ``overall.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import zip_longest
from multiprocessing import Pool
from pathlib import Path

import av
import numpy as np
from tqdm import tqdm


EPS32 = np.finfo(np.float32).eps


def _normalize_map(saliency: np.ndarray) -> np.ndarray:
    """Scale a saliency map to the range from zero to one."""
    saliency = np.asarray(saliency, dtype=np.float32)
    minimum = saliency.min()
    maximum = saliency.max()
    return (saliency - minimum) / (maximum - minimum + EPS32)


def _fixations_array(fixations, shape: tuple[int, int], frame_index: int) -> np.ndarray:
    """Validate one frame's fixations and return zero-based ``[y, x]`` coordinates."""
    points = np.asarray(fixations, dtype=np.int64)
    if points.size == 0:
        raise ValueError(f"Frame {frame_index}: expected at least one fixation")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Frame {frame_index}: fixations must have shape [N,2], got {points.shape}")
    height, width = shape
    invalid = (
        (points[:, 0] < 0)
        | (points[:, 0] >= height)
        | (points[:, 1] < 0)
        | (points[:, 1] >= width)
    )
    if invalid.any():
        raise ValueError(
            f"Frame {frame_index}: fixation {points[np.flatnonzero(invalid)[0]].tolist()} "
            f"is outside saliency map {width}x{height}"
        )
    return points


def nss(s_map: np.ndarray, fixations: np.ndarray) -> float:
    """Calculate normalized scanpath saliency at the fixation events."""
    if len(fixations) == 0:
        raise ValueError("NSS requires at least one fixation")
    normalized = (s_map - s_map.mean()) / (s_map.std() + EPS32)
    return float(normalized[fixations[:, 0], fixations[:, 1]].mean())


def similarity(s_map: np.ndarray, gt: np.ndarray) -> float:
    """Calculate histogram-intersection similarity between two saliency maps."""
    s_probability = s_map / (s_map.sum() + EPS32)
    gt_probability = gt / (gt.sum() + EPS32)
    return float(np.minimum(s_probability, gt_probability).sum())


def cc(s_map: np.ndarray, gt: np.ndarray) -> float:
    """Calculate the linear correlation coefficient between two saliency maps."""
    prediction = (s_map - s_map.mean()) / (s_map.std() + EPS32)
    target = (gt - gt.mean()) / (gt.std() + EPS32)
    denominator = np.sqrt((prediction ** 2).sum() * (target ** 2).sum() + EPS32)
    return float((prediction * target).sum() / denominator)


def auc_judd(s_map: np.ndarray, fixations: np.ndarray) -> float:
    """Calculate event-weighted AUC-Judd while preserving repeated fixations.

    Repeated ``[y, x]`` samples contribute separately to the true-positive
    rate. Only unique fixation
    pixels are removed from the uniform negative-pixel population.
    """
    if len(fixations) == 0:
        raise ValueError("AUC-Judd requires at least one fixation")

    # Used only to construct negative set
    unique_fixations = np.unique(fixations, axis=0)
    # Frames are decoded as gray16le. Re-quantizing the normalized map to uint16
    # preserves all source 10-bit levels while avoiding an H*W sort.
    quantized = np.rint(np.clip(s_map, 0.0, 1.0) * 65535.0).astype(np.uint16)
    event_values = quantized[fixations[:, 0], fixations[:, 1]]
    unique_fixation_values = quantized[unique_fixations[:, 0], unique_fixations[:, 1]]
    thresholds = np.unique(event_values)[::-1]
    histogram = np.bincount(quantized.reshape(-1), minlength=65536)
    pixels_at_or_above = np.cumsum(histogram[::-1], dtype=np.int64)[::-1]

    true_positives = np.empty(len(thresholds) + 2, dtype=np.float64)
    false_positives = np.empty(len(thresholds) + 2, dtype=np.float64)
    true_positives[0] = false_positives[0] = 0.0
    true_positives[-1] = false_positives[-1] = 1.0

    positive_count = len(event_values)
    # TODO: Check whether evaluation should adopt the revised MIT/Tuebingen
    # convention that samples nonfixations from all pixels (including fixation
    # pixels). It remains well-defined and approaches AUC=0.5 when fixations
    # become spatially uniform, unlike the shrinking negative set used here.
    negative_count = quantized.size - len(unique_fixations)
    if negative_count <= 0:
        return 1.0

    # Use duplicate positions for positive hits
    positive_hits = (
        event_values[:, None] >= thresholds[None, :]
    ).sum(axis=0)
    # Use unique positions for negative hits
    excluded_pixel_hits = (
        unique_fixation_values[:, None] >= thresholds[None, :]
    ).sum(axis=0)
    negative_hits = pixels_at_or_above[thresholds] - excluded_pixel_hits
    true_positives[1:-1] = positive_hits / positive_count
    false_positives[1:-1] = negative_hits / negative_count

    return float(np.trapezoid(true_positives, false_positives))


def calculate_frame_metrics(
    prediction: np.ndarray, gt: np.ndarray, fixations, frame_index: int
) -> dict[str, float]:
    """Validate and calculate all supported metrics for one video frame."""
    if prediction.shape != gt.shape:
        raise ValueError(
            f"Frame {frame_index}: prediction shape {prediction.shape} does not match "
            f"ground-truth shape {gt.shape}. Resize the predicted saliency video before "
            "running this benchmark, using the interpolation method appropriate for your model."
        )
    prediction = _normalize_map(prediction)
    gt = _normalize_map(gt)
    points = _fixations_array(fixations, gt.shape, frame_index)
    return {
        "sim": similarity(prediction, gt),
        "nss": nss(prediction, points),
        "cc": cc(prediction, gt),
        "auc_judd": auc_judd(prediction, points),
    }


def _configure_video_stream(container, decoder_threads: int):
    """Select a container's video stream and configure its decoding concurrency."""
    if not container.streams.video:
        raise ValueError(f"No video stream in {container.name}")
    stream = container.streams.video[0]
    stream.codec_context.thread_count = decoder_threads
    if decoder_threads > 1:
        stream.thread_type = "AUTO"
    return stream


def calculate_video_metrics(
    video_name: str,
    prediction_path: Path,
    gt_path: Path,
    fixation_path: Path,
    decoder_threads: int = 1,
) -> dict:
    """Evaluate corresponding frames from one prediction, target, and fixation file."""
    fixations = json.loads(fixation_path.read_text(encoding="utf-8"))
    sentinel = object()
    metric_lists = {"cc": [], "sim": [], "nss": [], "auc_judd": []}

    with av.open(str(prediction_path)) as prediction_container, av.open(str(gt_path)) as gt_container:
        prediction_stream = _configure_video_stream(prediction_container, decoder_threads)
        gt_stream = _configure_video_stream(gt_container, decoder_threads)
        prediction_frames = prediction_container.decode(prediction_stream)
        gt_frames = gt_container.decode(gt_stream)

        for frame_index, (prediction_frame, gt_frame, frame_fixations) in enumerate(
            zip_longest(prediction_frames, gt_frames, fixations, fillvalue=sentinel)
        ):
            if prediction_frame is sentinel or gt_frame is sentinel or frame_fixations is sentinel:
                raise ValueError(
                    f"{video_name}: frame-count mismatch near frame {frame_index}; "
                    "prediction, GT, and fixation JSON must have equal lengths"
                )
            # gray16le retains yuv420p10le precision; decoding to `gray` would
            # silently truncate both inputs to 8 bits.
            prediction = prediction_frame.to_ndarray(format="gray16le")
            gt = gt_frame.to_ndarray(format="gray16le")
            scores = calculate_frame_metrics(prediction, gt, frame_fixations, frame_index)
            for metric in ["cc", "sim", "nss", "auc_judd"]:
                metric_lists[metric].append(scores[metric])

    return {"video_name": video_name, **metric_lists}


def _video_map(directory: str | Path) -> dict[str, Path]:
    """Index the MP4 files in a directory by filename stem."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(root)
    videos: dict[str, Path] = {}
    for video_path in root.iterdir():
        if video_path.is_file() and video_path.suffix.lower() == ".mp4":
            if video_path.stem in videos:
                raise ValueError(f"Duplicate video stem {video_path.stem!r} under {root}")
            videos[video_path.stem] = video_path
    if not videos:
        raise ValueError(f"No MP4 videos found under {root}")
    return videos


def _process_video(task):
    """Evaluate one video task and save its per-frame metric results."""
    video_name, prediction_path, gt_path, fixation_path, result_path, decoder_threads = task
    result = calculate_video_metrics(
        video_name, prediction_path, gt_path, fixation_path, decoder_threads=decoder_threads
    )
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result), encoding="utf-8")
    os.replace(temporary, result_path)
    return video_name


def _write_overall(output_root: Path, video_names: list[str]) -> None:
    """Write frame-weighted and video-weighted means across the dataset."""
    metric_names = ("cc", "sim", "nss", "auc_judd")
    metric_values = {metric: [] for metric in metric_names}
    video_means = {metric: [] for metric in metric_names}

    for video_name in video_names:
        result_path = output_root / f"{video_name}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for metric in metric_names:
            values = np.asarray(result[metric], dtype=np.float64)
            if values.size == 0:
                raise ValueError(f"Cannot aggregate {metric}: {video_name} has no frames")
            if not np.isfinite(values).all():
                raise ValueError(f"Cannot aggregate non-finite {metric} values for {video_name}")
            metric_values[metric].extend(values)
            video_means[metric].append(float(values.mean()))

    frame_weighted = {}
    video_weighted = {}
    for metric in metric_names:
        values = np.asarray(metric_values[metric], dtype=np.float64)
        frame_weighted[metric] = float(values.mean())
        video_weighted[metric] = float(np.mean(video_means[metric]))

    overall = {
        "frame_weighted": frame_weighted,
        "video_weighted": video_weighted,
        "num_videos": len(video_names),
        "num_frames": len(metric_values["cc"]),
    }

    result_path = output_root / "overall.json"
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    temporary.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    os.replace(temporary, result_path)


def make_bench(
    model_predictions_path,
    gt_saliency_path,
    gt_fixations_path,
    results_path="results",
    num_workers=4,
    decoder_threads=1,
    overwrite=False,
):
    """Evaluate every matched dataset video and write per-video and overall results."""
    prediction_videos = _video_map(model_predictions_path)
    gt_videos = _video_map(gt_saliency_path)
    selected = sorted(set(prediction_videos) | set(gt_videos))
    missing_predictions = [name for name in selected if name not in prediction_videos]
    missing_gt = [name for name in selected if name not in gt_videos]
    if missing_predictions or missing_gt:
        raise FileNotFoundError(
            f"Prediction and GT video sets differ: "
            f"missing predictions={missing_predictions[:10]}, missing GT={missing_gt[:10]}"
        )

    fixation_root = Path(gt_fixations_path).expanduser()
    if not fixation_root.is_dir():
        raise NotADirectoryError(fixation_root)
    missing_fixations = [
        str(fixation_root / f"{video_name}.json")
        for video_name in selected
        if not (fixation_root / f"{video_name}.json").is_file()
    ]
    if missing_fixations:
        raise FileNotFoundError(f"Missing fixation JSON files: {missing_fixations[:10]}")

    output_root = Path(results_path).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for video_name in selected:
        result_path = output_root / f"{video_name}.json"
        if result_path.exists() and not overwrite:
            continue
        tasks.append(
            (
                video_name,
                prediction_videos[video_name],
                gt_videos[video_name],
                fixation_root / f"{video_name}.json",
                result_path,
                decoder_threads,
            )
        )

    print(f"Benchmarking {len(tasks)} MP4 pairs with {num_workers} worker(s); results: {output_root}")
    if num_workers == 1:
        for task in tqdm(tasks, unit="video"):
            _process_video(task)
    else:
        with Pool(num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(_process_video, tasks), total=len(tasks), unit="video"):
                pass
    _write_overall(output_root, selected)


def parse_args():
    """Parse command-line options for the video saliency benchmark."""
    parser = argparse.ArgumentParser(
        description="Evaluate saliency MP4 files against saliency maps and fixations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_predictions_path",
        default="./SampleSubmission-CenterPrior",
        help="Directory of predicted saliency MP4 files.",
    )
    parser.add_argument(
        "--gt_saliency_path",
        default="./SaliencyTest/Test",
        help="Directory of ground-truth saliency MP4 files.",
    )
    parser.add_argument(
        "--gt_fixations_path",
        default="./FixationsTest/Test",
        help="Directory of per-video fixation JSON files.",
    )
    parser.add_argument(
        "--results_path",
        default="./results",
        help="Directory for results JSONs.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of videos to evaluate concurrently.",
    )
    parser.add_argument(
        "--decoder_threads",
        type=int,
        default=1,
        help="Number of decoder threads used for each video stream.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recalculate videos whose result JSON files already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_bench(
        args.model_predictions_path,
        args.gt_saliency_path,
        args.gt_fixations_path,
        args.results_path,
        args.num_workers,
        args.decoder_threads,
        args.overwrite,
    )
