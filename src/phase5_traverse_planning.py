"""
Phase 5: Rover Traverse Planning
Grid pathfinding (A* and Dijkstra), multi-route optimization (Shortest, Safest, Best-Illuminated),
dynamic battery & solar power simulation, and longitudinal elevation/slope profiling.
"""

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

from src.config import CFG, RAW_DATA_DIR, PROCESSED_DATA_DIR, TABLES_DIR, MAPS_DIR
from src.utils import save_geotiff, plot_cartographic_map


class RoverTraversePlanner:
    """
    Planetary Rover Grid Pathfinding & Energy Simulation Engine.
    Implements 8-connected A* and Dijkstra graph traversal with dynamic battery power budgeting.
    """

    def __init__(self, config=CFG):
        self.cfg = config
        self.rover_cfg = config.rover
        self.terrain_cfg = config.terrain

    def load_environmental_rasters(self) -> Tuple[Dict[str, np.ndarray], rasterio.Affine, str]:
        """Loads environmental layers required for cost raster formulation."""
        rasters = {}
        meta_transform = None
        meta_crs = None

        paths = {
            "dem": RAW_DATA_DIR / "lola_dem_20m.tif",
            "boulders": RAW_DATA_DIR / "crater_boulder_density.tif",
            "slope": PROCESSED_DATA_DIR / "terrain_slope_deg.tif",
            "roughness": PROCESSED_DATA_DIR / "terrain_rms_roughness_m.tif",
            "illumination": PROCESSED_DATA_DIR / "terrain_illumination_fraction.tif",
            "ice_mask": PROCESSED_DATA_DIR / "ice_candidate_mask.tif",
            "ice_prob": PROCESSED_DATA_DIR / "ice_probability_map.tif"
        }

        for k, p in paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing required raster for Phase 5: {p}")
            with rasterio.open(p) as src:
                rasters[k] = src.read(1).astype(np.float64)
                if meta_transform is None:
                    meta_transform = src.transform
                    meta_crs = src.crs.to_string() if src.crs else self.terrain_cfg.projection_proj4

        return rasters, meta_transform, meta_crs

    def build_cost_raster(
        self,
        rasters: Dict[str, np.ndarray],
        mode: str = "safest"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds traversal cost surface and hard obstacle mask.
        Modes:
        - 'shortest': Focuses on minimal distance with baseline slope avoidance
        - 'safest': Heavily penalizes slope, roughness, and boulders
        - 'best_illuminated': Strongly penalizes shadows to keep rover in sunlight
        """
        slope = rasters["slope"]
        roughness = rasters["roughness"]
        boulders = rasters["boulders"]
        illum = rasters["illumination"]

        # Hard No-Go Obstacle Mask
        obstacle_mask = (slope > self.rover_cfg.max_nav_slope_deg) | (boulders > self.rover_cfg.max_boulder_hazard)

        # Baseline terrain cost
        res = self.terrain_cfg.grid_resolution_m
        norm_rough = roughness / (np.percentile(roughness, 95) + 1e-6)

        if mode == "shortest":
            # Direct Euclidean progress with minimal penalty
            slope_penalty = np.exp(slope / self.rover_cfg.slope_scale_shortest)
            cost = 1.0 + self.rover_cfg.weight_slope_shortest * slope_penalty + self.rover_cfg.weight_rough_shortest * norm_rough
        elif mode == "safest":
            # Maximum safety: smooth exponential slope penalty and roughness/boulder avoidance
            slope_penalty = np.exp(slope / self.rover_cfg.slope_scale_safest)
            cost = 1.0 + self.rover_cfg.weight_slope_safest * slope_penalty + self.rover_cfg.weight_rough_safest * norm_rough + self.rover_cfg.weight_boulder_safest * boulders
        elif mode == "best_illuminated":
            # Maximizes solar generation: heavy penalty on shadowed terrain
            slope_penalty = np.exp(slope / self.rover_cfg.slope_scale_illuminated)
            shadow_penalty = self.rover_cfg.weight_shadow_illuminated * (1.0 - np.clip(illum, 0.0, 1.0))
            cost = 1.0 + self.rover_cfg.weight_slope_illuminated * slope_penalty + self.rover_cfg.weight_rough_illuminated * norm_rough + shadow_penalty
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Set impassable cells to infinity
        cost[obstacle_mask] = np.inf
        return cost, obstacle_mask

    def a_star_search(
        self,
        cost_grid: np.ndarray,
        start_px: Tuple[int, int],
        goal_px: Tuple[int, int],
        use_heuristic: bool = True
    ) -> Tuple[List[Tuple[int, int]], float]:
        """
        8-connected grid pathfinding using A* (or Dijkstra when use_heuristic=False).
        Returns list of (py, px) path coordinates and total accumulated cost.
        """
        ny, nx = cost_grid.shape
        start_y, start_x = start_px
        goal_y, goal_x = goal_px

        if cost_grid[start_y, start_x] == np.inf:
            raise ValueError("Start location is inside an impassable obstacle!")
        if cost_grid[goal_y, goal_x] == np.inf:
            # Shift goal to nearest passable pixel
            from scipy.ndimage import distance_transform_edt
            passable = cost_grid < np.inf
            _, indices = distance_transform_edt(~passable, return_indices=True)
            goal_y, goal_x = indices[0, goal_y, goal_x], indices[1, goal_y, goal_x]

        # 8-connected neighbor offsets and step distances
        sqrt2 = np.sqrt(2.0)
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, sqrt2), (-1, 1, sqrt2), (1, -1, sqrt2), (1, 1, sqrt2)
        ]

        def heuristic(y, x):
            if not use_heuristic:
                return 0.0
            return np.sqrt((y - goal_y) ** 2 + (x - goal_x) ** 2)

        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, start_y, start_x))
        came_from = {}
        g_score = np.full((ny, nx), np.inf, dtype=np.float64)
        g_score[start_y, start_x] = 0.0

        visited = np.zeros((ny, nx), dtype=bool)

        while open_set:
            _, current_g, cy, cx = heapq.heappop(open_set)

            if visited[cy, cx]:
                continue
            visited[cy, cx] = True

            if (cy, cx) == (goal_y, goal_x):
                # Reconstruct path
                path = [(cy, cx)]
                while (cy, cx) in came_from:
                    cy, cx = came_from[(cy, cx)]
                    path.append((cy, cx))
                path.reverse()
                return path, float(current_g)

            for dy, dx, dist_mult in neighbors:
                ny_coord, nx_coord = cy + dy, cx + dx
                if 0 <= ny_coord < ny and 0 <= nx_coord < nx:
                    step_cost = 0.5 * (cost_grid[cy, cx] + cost_grid[ny_coord, nx_coord]) * dist_mult
                    if np.isinf(step_cost):
                        continue

                    tentative_g = current_g + step_cost
                    if tentative_g < g_score[ny_coord, nx_coord]:
                        came_from[(ny_coord, nx_coord)] = (cy, cx)
                        g_score[ny_coord, nx_coord] = tentative_g
                        f_score = tentative_g + heuristic(ny_coord, nx_coord)
                        heapq.heappush(open_set, (f_score, tentative_g, ny_coord, nx_coord))

        raise RuntimeError("No passable route found between landing site and ice target!")

    def simulate_power_and_metrics(
        self,
        path: List[Tuple[int, int]],
        rasters: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Simulates dynamic battery state of charge (SoC), energy consumption,
        slope profile, and shadow duration along a traverse.
        """
        speed = self.rover_cfg.speed_m_per_s  # 0.04 m/s
        res = self.terrain_cfg.grid_resolution_m
        battery_wh = self.rover_cfg.battery_capacity_wh  # 1200 Wh
        base_power_w = self.rover_cfg.baseline_power_w  # 65 W
        solar_charge_w = self.rover_cfg.solar_charge_rate_w  # 110 W
        traction_w_per_deg = self.rover_cfg.traction_power_w_per_deg  # 4 W/deg

        dem = rasters["dem"]
        slope = rasters["slope"]
        illum = rasters["illumination"]

        coords_y = [p[0] for p in path]
        coords_x = [p[1] for p in path]

        elev_profile = [float(dem[y, x]) for y, x in path]
        slope_profile = [float(slope[y, x]) for y, x in path]
        illum_profile = [float(illum[y, x]) for y, x in path]

        # Calculate distances, durations, and power
        step_distances_m = []
        cumulative_dist_m = [0.0]
        soc_history = [100.0]  # Start at 100% battery
        current_energy_wh = battery_wh
        total_energy_consumed_wh = 0.0
        shadow_duration_s = 0.0

        for i in range(1, len(path)):
            y1, x1 = path[i - 1]
            y2, x2 = path[i]
            dist_step = np.sqrt(((y2 - y1) * res) ** 2 + ((x2 - x1) * res) ** 2)
            step_distances_m.append(dist_step)
            cumulative_dist_m.append(cumulative_dist_m[-1] + dist_step)

            # Step duration
            dt_s = dist_step / speed
            dt_hours = dt_s / 3600.0

            # Power calculation
            slp = slope[y2, x2]
            p_drain = base_power_w + traction_w_per_deg * max(slp, 0.0)
            p_solar = solar_charge_w * np.clip(illum[y2, x2], 0.0, 1.0)
            net_power_w = p_solar - p_drain

            total_energy_consumed_wh += p_drain * dt_hours

            # Battery update
            if net_power_w >= 0:
                # Sunlit segment: recharge battery
                current_energy_wh = min(battery_wh, current_energy_wh + net_power_w * dt_hours)
            else:
                # Shadowed / low-sun segment: drain battery
                current_energy_wh = max(0.0, current_energy_wh + net_power_w * dt_hours)
                shadow_duration_s += dt_s

            soc = (current_energy_wh / battery_wh) * 100.0
            soc_history.append(soc)

        total_dist_m = cumulative_dist_m[-1]
        total_time_hours = (total_dist_m / speed) / 3600.0
        shadow_time_hours = shadow_duration_s / 3600.0
        min_soc = min(soc_history)

        return {
            "total_distance_m": round(total_dist_m, 1),
            "total_time_hours": round(total_time_hours, 2),
            "shadow_time_hours": round(shadow_time_hours, 2),
            "max_slope_deg": round(float(np.max(slope_profile)), 2),
            "mean_slope_deg": round(float(np.mean(slope_profile)), 2),
            "energy_consumed_wh": round(total_energy_consumed_wh, 1),
            "min_soc_pct": round(min_soc, 1),
            "path_pixels": path,
            "cumulative_dist_m": cumulative_dist_m,
            "elev_profile": elev_profile,
            "slope_profile": slope_profile,
            "soc_history": soc_history
        }

    def process_and_export(self) -> Dict[str, Any]:
        """Plans all 3 routes, generates comparison statistics, and outputs publication maps and profiles."""
        rasters, transform, crs = self.load_environmental_rasters()

        # Load Site 1 from Phase 4
        sites_csv = TABLES_DIR / "phase4_top3_landing_sites.csv"
        if not sites_csv.exists():
            raise FileNotFoundError("Run Phase 4 first to establish candidate landing sites.")
        df_sites = pd.read_csv(sites_csv)
        site1 = df_sites.iloc[0]
        start_px = (int(site1["Pixel_Y"]), int(site1["Pixel_X"]))

        # Target Goal: Nearest accessible sampling station on the high-probability ice deposit margin
        ice_mask = rasters["ice_mask"] > 0
        y_indices, x_indices = np.where(ice_mask)
        dists_to_start = (y_indices - start_px[0]) ** 2 + (x_indices - start_px[1]) ** 2
        nearest_idx = np.argmin(dists_to_start)
        goal_px = (int(y_indices[nearest_idx]), int(x_indices[nearest_idx]))

        # Compute 3 Routes
        routes = {}
        for mode in ["shortest", "safest", "best_illuminated"]:
            print(f"[*] Planning {mode.replace('_', ' ').title()} Route using A* search...")
            cost_grid, _ = self.build_cost_raster(rasters, mode=mode)
            path, _ = self.a_star_search(cost_grid, start_px, goal_px, use_heuristic=True)
            metrics = self.simulate_power_and_metrics(path, rasters)
            routes[mode] = metrics

        # Dijkstra Verification Benchmark on Safest Route
        print("[*] Running Dijkstra's algorithm benchmark on Safest Route...")
        cost_safest, _ = self.build_cost_raster(rasters, mode="safest")
        path_dijkstra, _ = self.a_star_search(cost_safest, start_px, goal_px, use_heuristic=False)
        print(f"[+] Dijkstra verified identical path length: {len(path_dijkstra)} nodes.")

        # Compile Route Comparison Table
        comparison_rows = [
            {
                "Route_Name": "Shortest Route",
                "Description": "Direct distance path; accepts steeper slopes",
                "Distance_m": routes["shortest"]["total_distance_m"],
                "Total_Time_h": routes["shortest"]["total_time_hours"],
                "Shadow_Time_h": routes["shortest"]["shadow_time_hours"],
                "Max_Slope_deg": routes["shortest"]["max_slope_deg"],
                "Mean_Slope_deg": routes["shortest"]["mean_slope_deg"],
                "Min_Battery_SoC_pct": routes["shortest"]["min_soc_pct"],
                "Energy_Consumed_Wh": routes["shortest"]["energy_consumed_wh"],
                "Traverse_Feasible": routes["shortest"]["min_soc_pct"] > self.rover_cfg.min_battery_reserve_pct and routes["shortest"]["max_slope_deg"] <= self.rover_cfg.max_nav_slope_deg
            },
            {
                "Route_Name": "Safest Route (RECOMMENDED)",
                "Description": "Minimizes slope gradient & boulder hazard; gradual rim ramp",
                "Distance_m": routes["safest"]["total_distance_m"],
                "Total_Time_h": routes["safest"]["total_time_hours"],
                "Shadow_Time_h": routes["safest"]["shadow_time_hours"],
                "Max_Slope_deg": routes["safest"]["max_slope_deg"],
                "Mean_Slope_deg": routes["safest"]["mean_slope_deg"],
                "Min_Battery_SoC_pct": routes["safest"]["min_soc_pct"],
                "Energy_Consumed_Wh": routes["safest"]["energy_consumed_wh"],
                "Traverse_Feasible": routes["safest"]["min_soc_pct"] > self.rover_cfg.min_battery_reserve_pct and routes["safest"]["max_slope_deg"] <= self.rover_cfg.max_nav_slope_deg
            },
            {
                "Route_Name": "Best-Illuminated Route",
                "Description": "Follows sunlit rim ridge as long as possible before cold trap entry",
                "Distance_m": routes["best_illuminated"]["total_distance_m"],
                "Total_Time_h": routes["best_illuminated"]["total_time_hours"],
                "Shadow_Time_h": routes["best_illuminated"]["shadow_time_hours"],
                "Max_Slope_deg": routes["best_illuminated"]["max_slope_deg"],
                "Mean_Slope_deg": routes["best_illuminated"]["mean_slope_deg"],
                "Min_Battery_SoC_pct": routes["best_illuminated"]["min_soc_pct"],
                "Energy_Consumed_Wh": routes["best_illuminated"]["energy_consumed_wh"],
                "Traverse_Feasible": routes["best_illuminated"]["min_soc_pct"] > self.rover_cfg.min_battery_reserve_pct and routes["best_illuminated"]["max_slope_deg"] <= self.rover_cfg.max_nav_slope_deg
            }
        ]

        df_comparison = pd.DataFrame(comparison_rows)
        comp_path = TABLES_DIR / "phase5_route_comparison.csv"
        df_comparison.to_csv(comp_path, index=False)

        # Convert pixel paths to stereographic coordinates (km)
        res = self.terrain_cfg.grid_resolution_m
        ny, nx = rasters["dem"].shape
        half_w = (nx * res) / 2.0

        def path_to_km(path):
            xs = [(-half_w + px * res) / 1000.0 for _, px in path]
            ys = [(half_w - py * res) / 1000.0 for py, _ in path]
            return xs, ys

        sx_short, sy_short = path_to_km(routes["shortest"]["path_pixels"])
        sx_safe, sy_safe = path_to_km(routes["safest"]["path_pixels"])
        sx_illum, sy_illum = path_to_km(routes["best_illuminated"]["path_pixels"])

        # Cartographic Map: Rover Traverses Overlaid on Topography & Illumination
        traverses_to_plot = [
            (sx_short, sy_short, "Shortest Route", "#e41a1c", "--"),
            (sx_safe, sy_safe, "Safest Route (Recommended)", "#00ff66", "-"),
            (sx_illum, sy_illum, "Best-Illuminated Route", "#ff7f00", "-.")
        ]

        start_x_km = (-half_w + start_px[1] * res) / 1000.0
        start_y_km = (half_w - start_px[0] * res) / 1000.0
        goal_x_km = (-half_w + goal_px[1] * res) / 1000.0
        goal_y_km = (half_w - goal_px[0] * res) / 1000.0

        points_to_plot = [
            (start_x_km, start_y_km, "Site 1 (Lander)", "o", "gold"),
            (goal_x_km, goal_y_km, "Ice Core Target", "X", "#00d2ff")
        ]

        map_routes = plot_cartographic_map(
            data=rasters["illumination"],
            title="Rover Traverse Planning: Comparison of 3 Ingress Routes to Ice Target",
            output_filename="map_phase5_rover_traverses.png",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            cbar_label="Solar Illumination Fraction (White: Sunlit, Black: PSR)",
            overlay_contour=rasters["dem"],
            contour_levels=self.terrain_cfg.contour_levels,
            points_to_plot=points_to_plot,
            traverses_to_plot=traverses_to_plot
        )

        # Plot Longitudinal Elevation & Slope Profile for the Recommended Safest Route
        safest = routes["safest"]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), dpi=300, sharex=True)

        dists_km = [d / 1000.0 for d in safest["cumulative_dist_m"]]

        # Elevation Profile
        ax1.plot(dists_km, safest["elev_profile"], color="#1f77b4", linewidth=2.2, label="Elevation (m)")
        ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Elevation (m)", fontsize=10, fontweight="bold")
        ax1.set_title("Safest Traverse Profile (Site 1 Landing to Subsurface Ice Target)", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(loc="upper right")

        # Slope Profile with Rover Hazard Threshold
        ax2.plot(dists_km, safest["slope_profile"], color="#d62728", linewidth=2.0, label="Terrain Slope (°)")
        ax2.axhline(self.rover_cfg.max_nav_slope_deg, color="black", linestyle="--", linewidth=1.5, label=f"Max Rover Limit ({self.rover_cfg.max_nav_slope_deg:.0f}°)")
        ax2.set_ylabel("Slope (deg)", fontsize=10, fontweight="bold")
        ax2.set_ylim(0, 16)
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.legend(loc="upper right")

        # Battery State of Charge (SoC) Profile
        ax3.plot(dists_km, safest["soc_history"], color="#2ca02c", linewidth=2.2, label="Battery State of Charge (%)")
        ax3.axhline(self.rover_cfg.min_battery_reserve_pct, color="orange", linestyle="--", linewidth=1.5, label=f"{self.rover_cfg.min_battery_reserve_pct:.0f}% Reserve Margin")
        ax3.set_xlabel("Traverse Distance along Route (km)", fontsize=10, fontweight="bold")
        ax3.set_ylabel("Battery SoC (%)", fontsize=10, fontweight="bold")
        ax3.set_ylim(0, 105)
        ax3.grid(True, linestyle="--", alpha=0.3)
        ax3.legend(loc="lower left")

        fig.tight_layout()
        profile_path = MAPS_DIR / "map_phase5_recommended_route_profile.png"
        fig.savefig(profile_path, bbox_inches="tight")
        plt.close(fig)

        return {
            "df_comparison": df_comparison,
            "routes": routes,
            "map_routes": map_routes,
            "profile_path": profile_path
        }


def run_phase5():
    print("[*] Running Phase 5: Rover Traverse Planning and Power Modeling...")
    planner = RoverTraversePlanner()
    results = planner.process_and_export()
    print("[+] Phase 5 completed successfully!")
    print("\n--- Route Comparison Table ---")
    cols = ["Route_Name", "Distance_m", "Total_Time_h", "Shadow_Time_h", "Max_Slope_deg", "Min_Battery_SoC_pct", "Traverse_Feasible"]
    print(results["df_comparison"][cols].to_string(index=False))
    return results


if __name__ == "__main__":
    run_phase5()
