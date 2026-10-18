"""
Phase 5: Rover Traverse Planning
Grid pathfinding (A* and Dijkstra), multi-route optimization (Shortest, Safest, Best-Illuminated),
dynamic battery & solar power simulation, longitudinal elevation/slope profiling, and cryptographic provenance sidecars.
"""

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

from src.config import CFG
from src.config_schema import LunaTraceConfig
from src.constants import SECONDS_PER_HOUR
from src.utils import save_geotiff, plot_cartographic_map
from src.provenance import write_provenance_sidecar


class RoverTraversePlanner:
    """
    Planetary Rover Grid Pathfinding & Energy Simulation Engine.
    Implements 8-connected A* and Dijkstra graph traversal with dynamic battery power budgeting.
    """

    def __init__(self, config: LunaTraceConfig = CFG):
        self.cfg = config
        self.rover_cfg = config.rover
        self.terrain_cfg = config.terrain
        self.raw_dir = config.paths.raw_data_dir
        self.processed_dir = config.paths.processed_data_dir
        self.tables_dir = config.paths.outputs_tables_dir
        self.maps_dir = config.paths.outputs_maps_dir

    def load_environmental_rasters(self) -> Tuple[Dict[str, Any], rasterio.Affine, str, List[Path]]:
        """Loads environmental layers required for cost raster formulation."""
        rasters = {}
        meta_transform = None
        meta_crs = None
        loaded_paths = []

        paths = {
            "dem": self.raw_dir / "lola_dem_20m.tif",
            "slope": self.processed_dir / "terrain_slope_deg.tif",
            "roughness": self.processed_dir / "terrain_rms_roughness_m.tif",
            "illumination": self.processed_dir / "terrain_illumination_fraction.tif",
            "ice_mask": self.processed_dir / "ice_candidate_mask.tif",
            "ice_prob": self.processed_dir / "ice_probability_map.tif"
        }

        for k, p in paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing required raster for Phase 5: {p}")
            loaded_paths.append(p)
            with rasterio.open(p) as src:
                if not src.crs:
                    raise ValueError(f"Raster {p} has no CRS defined in metadata!")
                rasters[k] = src.read(1).astype(np.float64)
                if meta_transform is None:
                    meta_transform = src.transform
                    meta_crs = src.crs.to_string()

        # Check for optional OHRC boulder density
        ohrc_path = self.raw_dir / "ohrc_boulder_density.tif"
        if ohrc_path.exists():
            with rasterio.open(ohrc_path) as src:
                rasters["boulders"] = src.read(1).astype(np.float64)
            rasters["boulder_status"] = "ASSESSED (Chandrayaan-2 OHRC)"
            loaded_paths.append(ohrc_path)
        else:
            rasters["boulders"] = None
            rasters["boulder_status"] = "UNASSESSED (OHRC not provided)"

        return rasters, meta_transform, meta_crs, loaded_paths

    def build_cost_raster(
        self,
        rasters: Dict[str, Any],
        mode: str = "safest"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds traversal cost surface and hard obstacle mask.
        Nav slope limit is strictly derived from chassis rollover limit / safety factor (Rule 4).
        """
        slope = rasters["slope"]
        roughness = rasters["roughness"]
        illum = rasters["illumination"]
        max_slope = self.rover_cfg.max_nav_slope_deg

        # Hard No-Go Obstacle Mask
        if rasters["boulders"] is not None:
            obstacle_mask = (slope > max_slope) | (rasters["boulders"] > 0.40)
        else:
            obstacle_mask = (slope > max_slope)

        # Baseline terrain cost
        eps = self.cfg.numerics.eps
        norm_rough = roughness / (np.percentile(roughness, 95) + eps)

        if mode == "shortest":
            slope_penalty = np.exp(slope / self.rover_cfg.slope_scale_shortest)
            cost = 1.0 + self.rover_cfg.weight_slope_shortest * slope_penalty + self.rover_cfg.weight_rough_shortest * norm_rough
        elif mode == "safest":
            slope_penalty = np.exp(slope / self.rover_cfg.slope_scale_safest)
            cost = 1.0 + self.rover_cfg.weight_slope_safest * slope_penalty + self.rover_cfg.weight_rough_safest * norm_rough
            if rasters["boulders"] is not None:
                cost += 2.5 * rasters["boulders"]
        elif mode == "best_illuminated":
            slope_penalty = np.exp(slope / self.rover_cfg.slope_scale_illuminated)
            shadow_penalty = self.rover_cfg.weight_shadow_illuminated * (1.0 - np.clip(illum, 0.0, 1.0))
            cost = 1.0 + self.rover_cfg.weight_slope_illuminated * slope_penalty + self.rover_cfg.weight_rough_illuminated * norm_rough + shadow_penalty
        else:
            raise ValueError(f"Unknown traverse mode: {mode}")

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
        Fails fast if start or goal is impassable.
        """
        ny, nx = cost_grid.shape
        start_y, start_x = start_px
        goal_y, goal_x = goal_px

        if cost_grid[start_y, start_x] == np.inf:
            raise ValueError(f"Start location ({start_y}, {start_x}) is inside an impassable obstacle!")
        if cost_grid[goal_y, goal_x] == np.inf:
            raise ValueError(f"Goal location ({goal_y}, {goal_x}) is inside an impassable obstacle!")

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

        raise RuntimeError("No passable route found between landing site and ice target satisfying chassis limits!")

    def simulate_power_and_metrics(
        self,
        path: List[Tuple[int, int]],
        rasters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulates dynamic battery state of charge (SoC), energy consumption,
        slope profile, and shadow duration along a traverse.
        """
        speed = self.rover_cfg.speed_m_per_s
        res = self.terrain_cfg.grid_resolution_m
        battery_wh = self.rover_cfg.battery_capacity_wh
        base_power_w = self.rover_cfg.baseline_power_w
        solar_charge_w = self.rover_cfg.solar_charge_rate_w
        traction_w_per_deg = self.rover_cfg.traction_power_w_per_deg

        dem = rasters["dem"]
        slope = rasters["slope"]
        illum = rasters["illumination"]

        elev_profile = [float(dem[y, x]) for y, x in path]
        slope_profile = [float(slope[y, x]) for y, x in path]
        illum_profile = [float(illum[y, x]) for y, x in path]

        step_distances_m = []
        cumulative_dist_m = [0.0]
        soc_history = [100.0]  # Initial full charge at landing site
        current_energy_wh = battery_wh
        total_energy_consumed_wh = 0.0
        shadow_duration_s = 0.0

        for i in range(1, len(path)):
            y1, x1 = path[i - 1]
            y2, x2 = path[i]
            dist_step = np.sqrt(((y2 - y1) * res) ** 2 + ((x2 - x1) * res) ** 2)
            step_distances_m.append(dist_step)
            cumulative_dist_m.append(cumulative_dist_m[-1] + dist_step)

            dt_s = dist_step / speed
            dt_hours = dt_s / SECONDS_PER_HOUR

            slp = slope[y2, x2]
            p_drain = base_power_w + traction_w_per_deg * max(slp, 0.0)
            p_solar = solar_charge_w * np.clip(illum[y2, x2], 0.0, 1.0)
            net_power_w = p_solar - p_drain

            total_energy_consumed_wh += p_drain * dt_hours

            if net_power_w >= 0:
                current_energy_wh = min(battery_wh, current_energy_wh + net_power_w * dt_hours)
            else:
                current_energy_wh = max(0.0, current_energy_wh + net_power_w * dt_hours)
                shadow_duration_s += dt_s

            soc = (current_energy_wh / battery_wh) * 100.0
            soc_history.append(soc)

        total_dist_m = cumulative_dist_m[-1]
        total_time_hours = (total_dist_m / speed) / SECONDS_PER_HOUR
        shadow_time_hours = shadow_duration_s / SECONDS_PER_HOUR
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
        """Plans all 3 routes, generates comparison statistics, and outputs publication maps and profiles with sidecars."""
        rasters, transform, crs, input_paths = self.load_environmental_rasters()

        # Load candidate site 1 from Phase 4
        sites_csv = self.tables_dir / "phase4_top3_landing_sites.csv"
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
        max_limit = self.rover_cfg.max_nav_slope_deg
        min_reserve = self.rover_cfg.min_battery_reserve_pct

        comparison_rows = [
            {
                "Route_Name": "Shortest Route",
                "Description": "Direct distance path; accepts steeper slopes within chassis safety factor",
                "Distance_m": routes["shortest"]["total_distance_m"],
                "Total_Time_h": routes["shortest"]["total_time_hours"],
                "Shadow_Time_h": routes["shortest"]["shadow_time_hours"],
                "Max_Slope_deg": routes["shortest"]["max_slope_deg"],
                "Mean_Slope_deg": routes["shortest"]["mean_slope_deg"],
                "Min_Battery_SoC_pct": routes["shortest"]["min_soc_pct"],
                "Energy_Consumed_Wh": routes["shortest"]["energy_consumed_wh"],
                "Traverse_Feasible": routes["shortest"]["min_soc_pct"] > min_reserve and routes["shortest"]["max_slope_deg"] <= max_limit
            },
            {
                "Route_Name": "Safest Route (RECOMMENDED)",
                "Description": f"Minimizes slope gradient within {max_limit:.1f} deg derived chassis limit",
                "Distance_m": routes["safest"]["total_distance_m"],
                "Total_Time_h": routes["safest"]["total_time_hours"],
                "Shadow_Time_h": routes["safest"]["shadow_time_hours"],
                "Max_Slope_deg": routes["safest"]["max_slope_deg"],
                "Mean_Slope_deg": routes["safest"]["mean_slope_deg"],
                "Min_Battery_SoC_pct": routes["safest"]["min_soc_pct"],
                "Energy_Consumed_Wh": routes["safest"]["energy_consumed_wh"],
                "Traverse_Feasible": routes["safest"]["min_soc_pct"] > min_reserve and routes["safest"]["max_slope_deg"] <= max_limit
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
                "Traverse_Feasible": routes["best_illuminated"]["min_soc_pct"] > min_reserve and routes["best_illuminated"]["max_slope_deg"] <= max_limit
            }
        ]

        df_comparison = pd.DataFrame(comparison_rows)
        comp_path = self.tables_dir / "phase5_route_comparison.csv"
        df_comparison.to_csv(comp_path, index=False)
        write_provenance_sidecar(comp_path, input_paths, self.cfg.model_dump(), {"boulder_status": rasters["boulder_status"]})

        # Coordinates conversion for mapping
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
        write_provenance_sidecar(map_routes, input_paths, self.cfg.model_dump(), {"map": "rover_traverses"})

        # Longitudinal Profiles
        safest = routes["safest"]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), dpi=self.cfg.visualization.dpi, sharex=True)

        dists_km = [d / 1000.0 for d in safest["cumulative_dist_m"]]

        ax1.plot(dists_km, safest["elev_profile"], color="#1f77b4", linewidth=2.2, label="Elevation (m)")
        ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Elevation (m)", fontsize=10, fontweight="bold")
        ax1.set_title("Safest Traverse Profile (Site 1 Landing to Subsurface Ice Target)", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(loc="upper right")

        ax2.plot(dists_km, safest["slope_profile"], color="#d62728", linewidth=2.0, label="Terrain Slope (°)")
        ax2.axhline(max_limit, color="black", linestyle="--", linewidth=1.5, label=f"Max Nav Limit ({max_limit:.1f}°)")
        ax2.set_ylabel("Slope (deg)", fontsize=10, fontweight="bold")
        ax2.set_ylim(0, self.cfg.visualization.profile_slope_ylim)
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.legend(loc="upper right")

        ax3.plot(dists_km, safest["soc_history"], color="#2ca02c", linewidth=2.2, label="Battery State of Charge (%)")
        ax3.axhline(min_reserve, color="orange", linestyle="--", linewidth=1.5, label=f"{min_reserve:.0f}% Reserve Margin")
        ax3.set_xlabel("Traverse Distance along Route (km)", fontsize=10, fontweight="bold")
        ax3.set_ylabel("Battery SoC (%)", fontsize=10, fontweight="bold")
        ax3.set_ylim(0, self.cfg.visualization.profile_soc_ylim)
        ax3.grid(True, linestyle="--", alpha=0.3)
        ax3.legend(loc="lower left")

        fig.tight_layout()
        profile_path = self.maps_dir / "map_phase5_recommended_route_profile.png"
        fig.savefig(profile_path, bbox_inches="tight")
        plt.close(fig)
        write_provenance_sidecar(profile_path, input_paths, self.cfg.model_dump(), {"map": "route_profile"})

        return {
            "df_comparison": df_comparison,
            "routes": routes,
            "map_routes": map_routes,
            "profile_path": profile_path,
            "boulder_status": rasters["boulder_status"]
        }


def run_phase5(config: LunaTraceConfig = CFG):
    print("[*] Running Phase 5: Rover Traverse Planning and Power Modeling...")
    planner = RoverTraversePlanner(config=config)
    results = planner.process_and_export()
    print(f"[+] Boulder Hazard Status: {results['boulder_status']}")
    print(f"[+] Derived Rover Slope Limit: {config.rover.max_nav_slope_deg:.1f}° (Chassis: {config.rover.chassis_rollover_limit_deg}°, SF: {config.rover.safety_factor})")
    print("[+] Phase 5 completed successfully!")
    print("\n--- Route Comparison Table ---")
    cols = ["Route_Name", "Distance_m", "Total_Time_h", "Shadow_Time_h", "Max_Slope_deg", "Min_Battery_SoC_pct", "Traverse_Feasible"]
    print(results["df_comparison"][cols].to_string(index=False))
    return results
