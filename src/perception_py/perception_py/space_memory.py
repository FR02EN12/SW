#!/usr/bin/env python3
import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String


class SpaceMemoryNode(Node):
    def __init__(self):
        super().__init__('space_memory')

        # Parameters
        self.declare_parameter('update_hz', 1.0)
        self.declare_parameter('search_radius_m', 3.0)
        self.declare_parameter('narrow_threshold_m', 0.40)
        self.declare_parameter('pullover_min_width_m', 0.50)
        self.declare_parameter('max_pullover_candidates', 10)

        self._update_hz = self.get_parameter('update_hz').value
        self._search_radius = self.get_parameter('search_radius_m').value
        self._narrow_thresh = self.get_parameter('narrow_threshold_m').value
        self._pullover_min_w = self.get_parameter('pullover_min_width_m').value
        self._max_candidates = self.get_parameter('max_pullover_candidates').value

        # State
        self._map_data: Optional[OccupancyGrid] = None
        self._map_grid: Optional[np.ndarray] = None
        self._map_resolution: float = 0.05
        self._map_origin_x: float = 0.0
        self._map_origin_y: float = 0.0
        self._map_width: int = 0
        self._map_height: int = 0

        self._robot_x: float = 0.0
        self._robot_y: float = 0.0
        self._robot_theta: float = 0.0

        self._narrow_passages: List[dict] = []
        self._pullover_candidates: List[dict] = []
        self._visited_cells: set = set()

        # QoS
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscriptions
        self.sub_map = self.create_subscription(
            OccupancyGrid, '/map', self._map_cb, map_qos)
        self.sub_pose = self.create_subscription(
            PoseStamped, '/pose', self._pose_cb, 10)

        # Publishers
        self.pub_pullover = self.create_publisher(
            String, '/memory/pull_over_candidates', 10)
        self.pub_narrow = self.create_publisher(
            String, '/memory/narrow_passages', 10)

        # Timer
        period = 1.0 / max(self._update_hz, 0.01)
        self.create_timer(period, self._update_cb)

        self.get_logger().info('SpaceMemoryNode initialised')

    # Subscription callbacks
    def _map_cb(self, msg: OccupancyGrid) -> None:
        self._map_data = msg
        self._map_resolution = msg.info.resolution
        self._map_origin_x = msg.info.origin.position.x
        self._map_origin_y = msg.info.origin.position.y
        self._map_width = msg.info.width
        self._map_height = msg.info.height
        self._map_grid = np.array(msg.data, dtype=np.int8).reshape(
            (self._map_height, self._map_width))

    def _pose_cb(self, msg: PoseStamped) -> None:
        self._robot_x = msg.pose.position.x
        self._robot_y = msg.pose.position.y
        q = msg.pose.orientation
        self._robot_theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # Record visited cell
        ci, cj = self._world_to_cell(self._robot_x, self._robot_y)
        self._visited_cells.add((ci, cj))

    # Coordinate helpers
    def _world_to_cell(self, wx: float, wy: float) -> Tuple[int, int]:
        ci = math.floor((wx - self._map_origin_x) / self._map_resolution)
        cj = math.floor((wy - self._map_origin_y) / self._map_resolution)
        return ci, cj

    def _cell_to_world(self, ci: int, cj: int) -> Tuple[float, float]:
        wx = (ci + 0.5) * self._map_resolution + self._map_origin_x
        wy = (cj + 0.5) * self._map_resolution + self._map_origin_y
        return wx, wy

    def _cell_in_bounds(self, ci: int, cj: int) -> bool:
        return 0 <= ci < self._map_width and 0 <= cj < self._map_height

    def _is_free(self, ci: int, cj: int) -> bool:
        if not self._cell_in_bounds(ci, cj):
            return False
        cell = int(self._map_grid[cj, ci])
        return 0 <= cell <= 50

    # Main analysis timer
    def _update_cb(self) -> None:
        if self._map_grid is None:
            return

        robot_ci, robot_cj = self._world_to_cell(self._robot_x, self._robot_y)
        radius_cells = int(self._search_radius / self._map_resolution)

        narrow_cells: List[Tuple[int, int, float]] = []
        wide_cells: List[Tuple[int, int, float]] = []

        # Scan nearby free cells and measure passage width
        ci_min = max(0, robot_ci - radius_cells)
        ci_max = min(self._map_width, robot_ci + radius_cells)
        cj_min = max(0, robot_cj - radius_cells)
        cj_max = min(self._map_height, robot_cj + radius_cells)

        for cj in range(cj_min, cj_max):
            for ci in range(ci_min, ci_max):
                if not self._is_free(ci, cj):
                    continue

                # Measure width: cast rays in 4 cardinal directions
                width_h = self._cast_width(ci, cj, dx=1, dy=0)
                width_v = self._cast_width(ci, cj, dx=0, dy=1)
                min_w = min(width_h, width_v) * self._map_resolution

                if min_w < self._narrow_thresh:
                    narrow_cells.append((ci, cj, min_w))
                elif min_w >= self._pullover_min_w:
                    wide_cells.append((ci, cj, min_w))

        # Cluster narrow cells into passage segments
        self._narrow_passages = self._cluster_narrow(narrow_cells)

        # Find pull-over candidates near narrow passages
        self._pullover_candidates = self._find_pullover(wide_cells)

        # Publish
        msg_narrow = String()
        msg_narrow.data = json.dumps(self._narrow_passages)
        self.pub_narrow.publish(msg_narrow)

        msg_pull = String()
        msg_pull.data = json.dumps(self._pullover_candidates)
        self.pub_pullover.publish(msg_pull)

    def _cast_width(self, ci: int, cj: int, dx: int, dy: int) -> int:
        """Cast in +/- direction and return total free cells span."""
        count = 1
        for sign in (1, -1):
            for step in range(1, 100):
                ni = ci + sign * step * dx
                nj = cj + sign * step * dy
                if self._is_free(ni, nj):
                    count += 1
                else:
                    break
        return count

    def _cluster_narrow(self, cells: List[Tuple[int, int, float]]) -> List[dict]:
        """Cluster nearby narrow cells into passage segments."""
        if not cells:
            return []

        cell_set = set((c[0], c[1]) for c in cells)
        visited = set()
        passages: List[dict] = []

        for ci, cj, w in cells:
            if (ci, cj) in visited:
                continue
            # BFS cluster
            cluster = []
            stack = [(ci, cj)]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                if cur not in cell_set:
                    continue
                visited.add(cur)
                cluster.append(cur)
                for dci, dcj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nb = (cur[0] + dci, cur[1] + dcj)
                    if nb not in visited and nb in cell_set:
                        stack.append(nb)

            if len(cluster) < 3:
                continue

            xs = [c[0] for c in cluster]
            ys = [c[1] for c in cluster]
            x1w, y1w = self._cell_to_world(min(xs), min(ys))
            x2w, y2w = self._cell_to_world(max(xs), max(ys))
            cluster_set = set(cluster)
            min_width = min((w for ci2, cj2, w in cells
                            if (ci2, cj2) in cluster_set), default=0.0)

            passages.append({
                'x1': round(x1w, 3), 'y1': round(y1w, 3),
                'x2': round(x2w, 3), 'y2': round(y2w, 3),
                'min_width_m': round(min_width, 3),
            })

        return passages

    def _find_pullover(self, wide_cells: List[Tuple[int, int, float]]) -> List[dict]:
        """Find pull-over candidates from wide cells near narrow passages."""
        if not wide_cells or not self._narrow_passages:
            return []

        candidates: List[dict] = []
        for ci, cj, width_m in wide_cells:
            wx, wy = self._cell_to_world(ci, cj)

            # Score by proximity to a narrow passage
            min_dist = float('inf')
            for passage in self._narrow_passages:
                pcx = (passage['x1'] + passage['x2']) / 2.0
                pcy = (passage['y1'] + passage['y2']) / 2.0
                d = math.hypot(wx - pcx, wy - pcy)
                min_dist = min(min_dist, d)

            if min_dist > self._search_radius:
                continue

            score = width_m / (1.0 + min_dist)
            candidates.append({
                'x': round(wx, 3),
                'y': round(wy, 3),
                # Use the robot's heading at observation time as the lane
                # direction so the path tracker aligns correctly on arrival.
                'theta': round(self._robot_theta, 4),
                'width_m': round(width_m, 3),
                'score': round(score, 3),
            })

        # Sort by score descending, keep top N
        candidates.sort(key=lambda c: c['score'], reverse=True)
        return candidates[:self._max_candidates]


def main(args=None):
    rclpy.init(args=args)
    node = SpaceMemoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except BaseException:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except BaseException:
                pass


if __name__ == '__main__':
    main()
