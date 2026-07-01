#!/usr/bin/env python3
"""
visual_analysis_suite.py
Unified script for:
- Single run rendering and metrics extraction
- Dual run comparison animation and metrics
- Topology metrics extraction
- GIF stitching
"""
import sys
import os
import numpy as np
import h5py  # type: ignore
import argparse
from PIL import Image
import imageio

try:
    import pyvista as pv
    from skimage.measure import marching_cubes, label
except ImportError:
    print("Error: Required libraries missing.")
    sys.exit(1)

DEFAULT_THRESHOLD_QUANTILE = 0.995
MIN_VOXEL_VOLUME = 8


def _compute_dynamic_threshold(rho, quantile=DEFAULT_THRESHOLD_QUANTILE):
    clamped_quantile = float(np.clip(quantile, 0.0, 1.0))
    return float(np.quantile(rho, clamped_quantile))


def _filter_small_components(mask, min_voxel_volume=MIN_VOXEL_VOLUME):
    labeled, num = label(mask, return_num=True, connectivity=1)
    if num == 0:
        return np.zeros_like(mask, dtype=bool), 0

    component_sizes = np.bincount(labeled.ravel())
    keep_labels = np.where(component_sizes >= int(max(1, min_voxel_volume)))[0]
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return np.zeros_like(mask, dtype=bool), 0

    filtered_mask = np.isin(labeled, keep_labels)
    return filtered_mask, int(keep_labels.size)


def _load_sse_history(sse_path, frame_count):
    if not sse_path:
        return np.zeros(frame_count, dtype=float)

    sse_history = np.load(sse_path)
    if sse_history.shape[0] < frame_count:
        padded = np.zeros(frame_count, dtype=float)
        padded[:sse_history.shape[0]] = sse_history
        return padded
    return sse_history[:frame_count]

# --- Mesh and Metrics Extraction ---
def extract_mesh_and_metrics(rho, quantile=DEFAULT_THRESHOLD_QUANTILE):
    threshold = _compute_dynamic_threshold(rho, quantile)
    mask = rho > threshold
    filtered_mask, component_count = _filter_small_components(mask)
    coverage = float(np.sum(filtered_mask)) / float(rho.size)

    if not np.any(filtered_mask):
        return None, 0.0, 0.0, coverage, threshold, component_count

    try:
        filtered_rho = np.where(filtered_mask, rho, np.nextafter(np.float64(threshold), -np.inf))
        verts, faces, normals, values = marching_cubes(filtered_rho, level=threshold)
        faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
        mesh = pv.PolyData(verts, faces)
        mesh["Elevation"] = verts[:, 2]
        return mesh, mesh.volume, mesh.area, coverage, threshold, component_count
    except Exception:
        return None, 0.0, 0.0, coverage, threshold, component_count

# --- Single Run Rendering ---
def render_single_run(rho_path, sse_path, run_name, output_gif, quantile=DEFAULT_THRESHOLD_QUANTILE):
    print(f"Rendering {run_name}...")
    with h5py.File(rho_path, "r") as f: history = f["rho_history"][:]
    T = history.shape[0]
    sse_history = _load_sse_history(sse_path, T)
    plotter = pv.Plotter(window_size=(800, 800), off_screen=True)
    plotter.set_background("black")
    plotter.open_gif(output_gif)
    metrics = {"coverage": [], "volume": [], "area": [], "threshold": [], "n_components": []}
    for t in range(T):
        plotter.clear_actors()
        rho_t = history[t]
        mesh, vol, area, coverage, threshold, component_count = extract_mesh_and_metrics(rho_t, quantile)
        if mesh is not None:
            plotter.add_mesh(mesh, scalars="Elevation", cmap="plasma", show_edges=True, edge_color="white", opacity=0.9)
            bounds = np.array([0, rho_t.shape[0], 0, rho_t.shape[1], 0, rho_t.shape[2]])
            plotter.reset_camera(bounds=bounds)
        plotter.render()
        metrics["coverage"].append(coverage)
        metrics["volume"].append(vol)
        metrics["area"].append(area)
        metrics["threshold"].append(threshold)
        metrics["n_components"].append(component_count)
        plotter.add_text(
            f"{run_name}\nTime: {t}\nCov: {coverage:.4f}\nComp: {component_count}\nThr: {threshold:.6g}\nVol: {vol:.0f}\nSSE: {sse_history[t]:.3f}",
            position="upper_left",
            font_size=12,
            color="white",
            name="hud",
        )
        plotter.write_frame()
        print(f"  Frame {t}/{T-1} rendered.")
    plotter.close()
    np.savez(output_gif.replace('.gif', '_metrics.npz'), **metrics)
    print(f"SUCCESS! Render and metrics saved for {run_name}.")

# --- Dual Run Comparison ---
def render_comparison_animation(rho_a, sse_a, rho_b, sse_b, output_gif, quantile=DEFAULT_THRESHOLD_QUANTILE):
    print("Loading datasets for comparison...")
    with h5py.File(rho_a, "r") as f: history_a = f["rho_history"][:]
    with h5py.File(rho_b, "r") as f: history_b = f["rho_history"][:]
    sse_history_a = np.load(sse_a)
    sse_history_b = np.load(sse_b)
    T = min(history_a.shape[0], history_b.shape[0])
    plotter = pv.Plotter(shape=(1, 3), window_size=(2400, 800), off_screen=True)
    plotter.open_gif(output_gif)
    plotter.subplot(0, 0); plotter.set_background("black")
    plotter.subplot(0, 1); plotter.set_background("black")
    plotter.subplot(0, 2); plotter.set_background("white")
    chart = pv.Chart2D()
    time_steps = np.arange(T)
    chart.line(time_steps, sse_history_a[:T], color='b', width=2.0, label="Run A")
    chart.line(time_steps, sse_history_b[:T], color='g', width=2.0, label="Run B")
    tracker_a = chart.scatter(np.array([0]), np.array([sse_history_a[0]]), color='blue', size=15)
    tracker_b = chart.scatter(np.array([0]), np.array([sse_history_b[0]]), color='green', size=15)
    chart.x_axis.title = "Time Step"
    chart.y_axis.title = "Spectral SSE"
    plotter.add_chart(chart)
    metrics = {
        "cov_a": [], "cov_b": [], "vol_a": [], "vol_b": [], "area_a": [], "area_b": [],
        "threshold_a": [], "threshold_b": [], "n_components_a": [], "n_components_b": []
    }
    print(f"Generating comparison animation ({T} frames)...")
    for t in range(T):
        plotter.subplot(0, 2)
        tracker_a.update(np.array([t]), np.array([sse_history_a[t]]))
        tracker_b.update(np.array([t]), np.array([sse_history_b[t]]))
        plotter.subplot(0, 0)
        plotter.clear_actors()
        rho_a_t = history_a[t]
        mesh_a, vol_a, area_a, cov_a, threshold_a, components_a = extract_mesh_and_metrics(rho_a_t, quantile)
        if mesh_a is not None:
            plotter.add_mesh(mesh_a, scalars="Elevation", cmap="plasma", show_edges=True, edge_color="white", opacity=0.9)
            bounds = np.array([0, rho_a_t.shape[0], 0, rho_a_t.shape[1], 0, rho_a_t.shape[2]])
            plotter.reset_camera(bounds=bounds)
        plotter.subplot(0, 1)
        plotter.clear_actors()
        rho_b_t = history_b[t]
        mesh_b, vol_b, area_b, cov_b, threshold_b, components_b = extract_mesh_and_metrics(rho_b_t, quantile)
        if mesh_b is not None:
            plotter.add_mesh(mesh_b, scalars="Elevation", cmap="viridis", show_edges=True, edge_color="white", opacity=0.9)
            bounds = np.array([0, rho_b_t.shape[0], 0, rho_b_t.shape[1], 0, rho_b_t.shape[2]])
            plotter.reset_camera(bounds=bounds)
        plotter.render()
        metrics["cov_a"].append(cov_a); metrics["cov_b"].append(cov_b)
        metrics["vol_a"].append(vol_a); metrics["vol_b"].append(vol_b)
        metrics["area_a"].append(area_a); metrics["area_b"].append(area_b)
        metrics["threshold_a"].append(threshold_a); metrics["threshold_b"].append(threshold_b)
        metrics["n_components_a"].append(components_a); metrics["n_components_b"].append(components_b)
        plotter.subplot(0, 0)
        plotter.add_text(
            f"RUN A\nCov: {cov_a:.4f}\nComp: {components_a}\nThr: {threshold_a:.6g}\nVol: {vol_a:.0f}\nSSE: {sse_history_a[t]:.3f}",
            position="upper_left", font_size=12, color="white", name="hud_a"
        )
        plotter.subplot(0, 1)
        plotter.add_text(
            f"RUN B\nCov: {cov_b:.4f}\nComp: {components_b}\nThr: {threshold_b:.6g}\nVol: {vol_b:.0f}\nSSE: {sse_history_b[t]:.3f}",
            position="upper_left", font_size=12, color="white", name="hud_b"
        )
        plotter.write_frame()
        print(f"  Frame {t}/{T-1} rendered.")
    plotter.close()
    np.savez(output_gif.replace('.gif', '_metrics.npz'), **metrics)
    print(f"\nSUCCESS! Comparison animation and metrics saved.")

# --- Topology Metrics Extraction ---
def compute_topology_metrics(rho_path, quantile=DEFAULT_THRESHOLD_QUANTILE, output_npz=None):
    print(f"Loading: {rho_path}")
    with h5py.File(rho_path, "r") as f:
        rho_history = f["rho_history"][:]
    T = rho_history.shape[0]
    n_components = np.zeros(T, dtype=int)
    surface_area = np.zeros(T)
    volume = np.zeros(T)
    coverage = np.zeros(T)
    thresholds = np.zeros(T)
    for t in range(T):
        rho = rho_history[t]
        mesh, mesh_volume, mesh_area, coverage_t, threshold_t, component_count = extract_mesh_and_metrics(rho, quantile)
        n_components[t] = component_count
        surface_area[t] = mesh_area
        volume[t] = mesh_volume
        coverage[t] = coverage_t
        thresholds[t] = threshold_t
        print(f"Frame {t}: threshold={thresholds[t]:.6g}, coverage={coverage[t]:.4f}, components={n_components[t]}, area={surface_area[t]:.2f}, volume={volume[t]:.2f}")
    if output_npz:
        np.savez(output_npz, n_components=n_components, surface_area=surface_area, volume=volume, coverage=coverage, thresholds=thresholds)
        print(f"Saved metrics to {output_npz}")
    return n_components, surface_area, volume

# --- GIF Stitching ---
def stitch_gifs_side_by_side(gif_paths, output_path):
    # Use imageio.get_reader for streaming frames
    readers = [imageio.get_reader(gif_path) for gif_path in gif_paths]
    n_frames = min(reader.get_length() for reader in readers)
    # Get base size from first frame
    first_frame = readers[0].get_data(0)
    base_size = (first_frame.shape[1], first_frame.shape[0])
    stitched_frames = []
    for i in range(n_frames):
        pil_frames = []
        for reader in readers:
            frame = reader.get_data(i)
            pil_img = Image.fromarray(frame)
            pil_img = pil_img.resize(base_size)
            pil_frames.append(pil_img)
        total_width = base_size[0] * len(pil_frames)
        new_img = Image.new('RGB', (total_width, base_size[1]))
        for idx, frame in enumerate(pil_frames):
            new_img.paste(frame, (idx * base_size[0], 0))
        stitched_frames.append(new_img)
    stitched_frames[0].save(output_path, save_all=True, append_images=stitched_frames[1:], duration=100, loop=0)
    print(f"SUCCESS! Stitched GIF saved to: {output_path}")

# --- Main CLI ---
def main():
    if "--input" in sys.argv and "--output" in sys.argv and len(sys.argv) > 1 and sys.argv[1] != "topology":
        legacy_parser = argparse.ArgumentParser(description="Legacy single-run renderer compatibility mode.")
        legacy_parser.add_argument('--input', required=True, help='Path to rho_history.h5')
        legacy_parser.add_argument('--output', required=True, help='Output GIF name')
        legacy_parser.add_argument('--sse', help='Optional path to sse_history.npy')
        legacy_parser.add_argument('--name', default='Simulation Run', help='Name overlay')
        legacy_parser.add_argument('--quantile', type=float, default=DEFAULT_THRESHOLD_QUANTILE, help='Quantile used for dynamic topology extraction.')
        legacy_args = legacy_parser.parse_args()
        render_single_run(legacy_args.input, legacy_args.sse, legacy_args.name, legacy_args.output, legacy_args.quantile)
        return

    parser = argparse.ArgumentParser(description="Unified visual analysis suite.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    # Single run
    p_single = subparsers.add_parser("single", help="Render single run animation and metrics.")
    p_single.add_argument('--rho', required=True, help='Path to rho_history.h5')
    p_single.add_argument('--sse', required=True, help='Path to sse_history.npy')
    p_single.add_argument('--name', default='Simulation Run', help='Name overlay')
    p_single.add_argument('--out', default='render.gif', help='Output GIF name')
    p_single.add_argument('--quantile', type=float, default=DEFAULT_THRESHOLD_QUANTILE, help='Quantile used for dynamic topology extraction.')
    # Dual comparison
    p_dual = subparsers.add_parser("compare", help="Render dual run comparison animation and metrics.")
    p_dual.add_argument('--rho_a', required=True, help='Run A rho_history.h5')
    p_dual.add_argument('--sse_a', required=True, help='Run A sse_history.npy')
    p_dual.add_argument('--rho_b', required=True, help='Run B rho_history.h5')
    p_dual.add_argument('--sse_b', required=True, help='Run B sse_history.npy')
    p_dual.add_argument('--out', default='basin_comparison.gif', help='Output GIF name')
    p_dual.add_argument('--quantile', type=float, default=DEFAULT_THRESHOLD_QUANTILE, help='Quantile used for dynamic topology extraction.')
    # Topology metrics
    p_topo = subparsers.add_parser("topology", help="Extract topology metrics from run.")
    p_topo.add_argument('--input', required=True, help='Path to rho_history.h5')
    p_topo.add_argument('--quantile', type=float, default=DEFAULT_THRESHOLD_QUANTILE, help='Quantile used for dynamic topology extraction.')
    p_topo.add_argument('--output', help='Output .npz file for metrics')
    # GIF stitching
    p_stitch = subparsers.add_parser("stitch", help="Stitch GIFs side-by-side.")
    p_stitch.add_argument('--gifs', nargs='+', required=True, help='List of GIFs to stitch')
    p_stitch.add_argument('--out', required=True, help='Output GIF name')
    args = parser.parse_args()
    if args.mode == "single":
        render_single_run(args.rho, args.sse, args.name, args.out, args.quantile)
    elif args.mode == "compare":
        render_comparison_animation(args.rho_a, args.sse_a, args.rho_b, args.sse_b, args.out, args.quantile)
    elif args.mode == "topology":
        compute_topology_metrics(args.input, args.quantile, args.output)
    elif args.mode == "stitch":
        stitch_gifs_side_by_side(args.gifs, args.out)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
