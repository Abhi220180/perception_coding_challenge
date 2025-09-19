#!/usr/bin/env python3
""" main script for wisconsin autonomous perception coding challenge. """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
import os
from pathlib import Path
import cv2
from typing import List, Tuple, Optional


class TrajectoryEstimator:
    def __init__(self, dataset_path: str):
        """ initialize the trajectory estimator. """
        self.dataset_path = Path(dataset_path)
        self.rgb_path = self.dataset_path / "rgb"
        self.xyz_path = self.dataset_path / "xyz"
        self.bbox_csv_path = self.dataset_path / "bbox_light.csv"

        self.bbox_data = self._load_bbox_data()
        self.trajectory_points = []
        self.traffic_light_positions = []
        
    def _load_bbox_data(self) -> pd.DataFrame:
        """ load and clean bounding box csv data. """
        df = pd.read_csv(self.bbox_csv_path)
        
        # filter invalid boxes.
        valid_mask = (df[['x1', 'y1', 'x2', 'y2']] != 0).any(axis=1)
        df = df[valid_mask].copy()
        
        return df
    
    def get_traffic_light_center(self, frame_id: int) -> Optional[Tuple[int, int]]:
        """ get traffic light center pixel from bounding box. """
        frame_data = self.bbox_data[self.bbox_data['frame'] == frame_id]
        
        if frame_data.empty:
            return None
            
        row = frame_data.iloc[0]
        u = int((row['x1'] + row['x2']) / 2)
        v = int((row['y1'] + row['y2']) / 2)
        
        return (u, v)
    
    def load_xyz_data(self, frame_id: int) -> Optional[np.ndarray]:
        """ load xyz point cloud data for a frame. """
        xyz_filename = f"depth{frame_id:06d}.npz"
        xyz_filepath = self.xyz_path / xyz_filename
        
        if not xyz_filepath.exists():
            return None
            
        try:
            data = np.load(xyz_filepath)
            # load xyz data, using only the first 3 channels.
            key = 'xyz' if 'xyz' in data.files else 'points'
            arr = data[key]

            if arr.ndim != 3 or arr.shape[2] < 3:
                print(f"Error: Unexpected data shape {arr.shape} in {xyz_filename}")
                return None

            return arr[..., :3].astype(np.float32)
        except Exception as e:
            print(f"Error loading {xyz_filename}: {e}")
            return None
    
    def get_3d_position(self, frame_id: int, u: int, v: int) -> Optional[Tuple[float, float, float]]:
        """ get 3d position from xyz data using a median of a local patch. """
        xyz_data = self.load_xyz_data(frame_id)
        if xyz_data is None:
            return None
        H, W, _ = xyz_data.shape
        if v < 0 or u < 0 or v >= H or u >= W:
            return None
        # for robustness, take the median of a 5x5 patch around the pixel.
        half = 2
        v0, v1 = max(0, v - half), min(H, v + half + 1)
        u0, u1 = max(0, u - half), min(W, u + half + 1)
        patch = xyz_data[v0:v1, u0:u1].reshape(-1, 3)

        valid_mask = (~np.isnan(patch).any(axis=1)) & (np.linalg.norm(patch, axis=1) > 1e-6)
        valid_points = patch[valid_mask]

        if valid_points.shape[0] == 0:
            return None

        return tuple(np.median(valid_points, axis=0))
    
    def compute_ground_frame_trajectory(self) -> List[Tuple[float, float]]:
        """ compute the ego-vehicle trajectory in the ground frame. """
        trajectory = []
        
        valid_frames = sorted(self.bbox_data['frame'].unique())
        
        if len(valid_frames) == 0:
            print("No valid frames found in bounding box data")
            return trajectory
            
        # establish world frame origin from the first valid frame.
        reference_frame = valid_frames[0]
        ref_center = self.get_traffic_light_center(reference_frame)
        
        if ref_center is None:
            print("No traffic light found in reference frame")
            return trajectory
            
        ref_tl_pos = self.get_3d_position(reference_frame, ref_center[0], ref_center[1])
        if ref_tl_pos is None:
            print(f"Error: Could not determine a valid 3D position for the traffic light in the reference frame ({reference_frame}).")
            print("Cannot establish ground frame origin. Aborting.")
            return trajectory
            
        print(f"Reference frame: {reference_frame}")
        print(f"Reference traffic light position (camera frame): {ref_tl_pos}")
        
        ref_X, ref_Y, ref_Z = ref_tl_pos
        
        total_frames = len(valid_frames)
        for idx, frame_id in enumerate(valid_frames):
            if idx % 10 == 0 or idx == total_frames - 1:
                print(f"Processing frame {idx + 1}/{total_frames} (ID: {frame_id})")

            center = self.get_traffic_light_center(frame_id)
            if center is None:
                continue

            tl_pos = self.get_3d_position(frame_id, center[0], center[1])
            if tl_pos is None:
                continue
                
            X, Y, Z = tl_pos
            
            # convert traffic light's apparent motion to ego-vehicle's motion.
            
            ground_x = -X
            ground_y = -Y
            
            trajectory.append((ground_x, ground_y))
            self.traffic_light_positions.append((X, Y, Z))
            
        print(f"Computed {len(trajectory)} trajectory points")
        return trajectory

    def create_trajectory_animation(self, trajectory: List[Tuple[float, float]], mp4_path: str = "trajectory.mp4", gif_path: str = "trajectory.gif", fps: int = 15):
        """ create an animated bev trajectory video. """
        if not trajectory:
            print("No trajectory points to animate")
            return
        traj = np.array(trajectory)
        x, y = traj[:, 0], traj[:, 1]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlabel('X (meters) - Forward from traffic light')
        ax.set_ylabel('Y (meters) - Left from traffic light')
        ax.set_title("Ego-Vehicle Trajectory in Ground Frame (Animated)")
        ax.grid(True, alpha=0.3)
        ax.scatter(0, 0, c='orange', s=150, marker='*', label='Traffic light (origin)', zorder=5)
        ax.legend()
        ax.set_aspect('equal', adjustable='datalim')
        ax.plot([], [])
        line, = ax.plot([], [], 'b-', lw=2, alpha=0.9)
        pts = ax.scatter([], [], c='blue', s=20, alpha=0.7)

        pad = 2.0
        ax.set_xlim(min(x) - pad, max(x) + pad)
        ax.set_ylim(min(y) - pad, max(y) + pad)

        def init():
            line.set_data([], [])
            pts.set_offsets(np.empty((0, 2)))
            return line, pts

        def update(i):
            line.set_data(x[:i+1], y[:i+1])
            pts.set_offsets(np.column_stack([x[:i+1], y[:i+1]]))
            return line, pts

        ani = animation.FuncAnimation(fig, update, frames=len(x), init_func=init, interval=1000/max(1, fps), blit=True)
        saved = False
        # try to save as mp4, fall back to gif if ffmpeg is unavailable.
        try:
            writer = animation.writers['ffmpeg'](fps=fps, metadata=dict(artist='Cascade'), bitrate=1800)
            ani.save(mp4_path, writer=writer)
            print(f"animation saved to {mp4_path}")
        except (RuntimeError, KeyError):
            warnings.warn(f"ffmpeg not available. Attempting to save as GIF at {gif_path}.")
            try:
                ani.save(gif_path, writer='pillow', fps=fps)
                print(f"Animation saved to {gif_path}")
            except Exception as e:
                warnings.warn(f"Could not save animation as GIF: {e}")
        plt.close(fig)
    
    def plot_trajectory(self, trajectory: List[Tuple[float, float]], output_path: str = "trajectory.png"):
        """ plot the ego-vehicle trajectory in bird's-eye view. """
        if not trajectory:
            print("No trajectory points to plot")
            return
            
        traj_array = np.array(trajectory)
        x_coords = traj_array[:, 0]
        y_coords = traj_array[:, 1]
        
        plt.figure(figsize=(12, 8))
        
        plt.scatter(x_coords, y_coords, c='blue', s=20, alpha=0.7, label='Ego-vehicle Trajectory')
        
        plt.plot(x_coords, y_coords, 'b-', alpha=0.5, linewidth=1)
        
        plt.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', label='Start', zorder=5)
        plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='s', label='End', zorder=5)
        
        plt.scatter(0, 0, c='orange', s=150, marker='*', label='Traffic Light (Origin)', zorder=5)
        
        plt.xlabel('X (meters) - Forward from Traffic Light')
        plt.ylabel('Y (meters) - Left from Traffic Light')
        plt.title("Ego-Vehicle Trajectory in Ground Frame (Bird's-Eye View)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal')
        
        total_distance = np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
        stats_text = f'Total Distance: {total_distance:.2f}m\nData Points: {len(trajectory)}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Trajectory plot saved to {output_path}")
        
    def run_full_pipeline(self):
        """ execute the full trajectory estimation and visualization pipeline. """
        print("Starting trajectory estimation pipeline...")

        print("\nStep 1: Computing ego-vehicle trajectory...")
        trajectory = self.compute_ground_frame_trajectory()
        if not trajectory:
            print("Trajectory computation failed. No valid points were generated.")
            return

        print("\nStep 2: Generating static trajectory plot...")
        self.plot_trajectory(trajectory)

        print("\nStep 3: Generating animated trajectory video...")
        self.create_trajectory_animation(trajectory)

        print("\n--- Trajectory Summary ---")
        traj_array = np.array(trajectory)
        total_distance = np.sum(np.sqrt(np.diff(traj_array[:, 0])**2 + np.diff(traj_array[:, 1])**2))
        print(f"  - Number of points: {len(trajectory)}")
        print(f"  - Total distance:   {total_distance:.2f} meters")
        print(f"  - X Range:          {traj_array[:, 0].min():.2f}m to {traj_array[:, 0].max():.2f}m")
        print(f"  - Y Range:          {traj_array[:, 1].min():.2f}m to {traj_array[:, 1].max():.2f}m")
        
        print("\nPipeline completed successfully!")


def main():
    """ main entry point. """
    script_dir = Path(__file__).parent
    dataset_path = script_dir / "dataset"

    if not dataset_path.exists() or not (dataset_path / 'rgb').exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure the 'dataset' folder is in the same directory as the script.")
        return

    estimator = TrajectoryEstimator(dataset_path)
    estimator.run_full_pipeline()

if __name__ == "__main__":
    main()
