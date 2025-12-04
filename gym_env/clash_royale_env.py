import gymnasium as gym
import numpy as np
import pygame
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from troop.archer import Archer
from troop.tower import Tower
from troop.troop import Troop
from troop.skeleton import Skeleton

color_map = {"Archer": (179, 0, 255),
             "Skeleton": (255,255,255),
             "Troop": (0,0,0)}

class ClashRoyaleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, max_steps: int = 10000, enemy_spawn_every: int = 500):
        super().__init__()
        self.rows = 32
        self.cols = 18
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)

        self.max_steps = max_steps
        self.enemy_spawn_every = enemy_spawn_every

        # Observation: simple integer grid with towers (+/-2) and troops (+/-1)
        self.observation_space = gym.spaces.Box(
            low=-2,
            high=2,
            shape=self.grid.shape,
            dtype=np.int8,
        )

        # Action: 0 = no-op, 1..(rows*cols) = try to spawn a friendly troop at that cell
        self.action_space = gym.spaces.Discrete(self.rows * self.cols + 1)

        # Rendering attributes
        self.tile_size = 25
        self.width = self.cols * self.tile_size
        self.height = self.rows * self.tile_size
        self.screen = None
        self.clock = None
        self.initialized = False
        self.font = None
        self.river_row = self.rows // 2  # middle river row for rendering/spawn blocking
        bridge_left = max(1, self.cols // 4)
        bridge_right = max(1, self.cols - self.cols // 4 - 1)
        # Make bridges two tiles wide by including the neighbor column on each side
        self.bridge_cols = {
            bridge_left,
            min(self.cols - 1, bridge_left + 1),
            bridge_right,
            min(self.cols - 1, bridge_right + 1),
        }
        self.bridge_tol = 0.3  # how close to bridge column to allow crossing
        self.river_block_band = 0.45  # how far from river center counts as blocked
        self.grass_dark = (115, 186, 78)
        self.grass_light = (141, 206, 101)
        self.path_color = (200, 175, 120)
        self.tower_friendly_color = (40, 110, 200)
        self.tower_enemy_color = (200, 80, 80)
        self.king_size_tiles = 2.2  # draw size, visual only

        # Game state
        self.troops: list[Troop] = []
        self.enemy_tower: Tower | None = None
        self.player_tower: Tower | None = None
        self.step_count = 0
        self.rng = np.random.default_rng()
        self._next_id = 0
        self.sim_substeps = 4  # smaller dt increments per env step to smooth motion

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seed(seed)
        self.grid[:] = 0
        self.troops = []
        self._init_towers()
        self.step_count = 0
        self._next_id = 0
        self._place_towers()
        observation = self._update_grid()
        return observation, {}

    def _spawn_random_friendly(self, row: int, col: int):
        """Spawn a random friendly troop at the specified cell."""
        troop_choices = [Troop, Archer, Skeleton]  # Extend here with more troop classes
        idx = int(self.rng.integers(0, len(troop_choices)))
        troop_cls = troop_choices[idx]
        self._spawn_troop(friendly=True, row=row, col=col, troop_cls=troop_cls)

    def _init_towers(self):
        center_col = self.cols // 2
        self.enemy_tower = Tower(
            friendly=False,
            x=0.0,
            y=float(center_col),
            hp=4500,
            attack_speed=20,  # slower cadence so troops can approach
            attack_range=7,  # outranged by archers to make chip damage possible
            attack_damage=58,
        )
        self.player_tower = Tower(
            friendly=True,
            x=float(self.rows - 1),
            y=float(center_col),
            hp=4500,
            attack_speed=20,
            attack_range=7,
            attack_damage=58,
        )

    def step(self, action):
        self.step_count += 1
        reward = -0.01  # small step penalty
        done = False

        self._handle_player_action(action)
        self._maybe_spawn_enemy()

        prev_enemy_hp = self.enemy_tower.hp if self.enemy_tower else 0
        prev_player_hp = self.player_tower.hp if self.player_tower else 0

        # Advance simulation by one tick
        dt = 1.0 / self.sim_substeps
        for _ in range(self.sim_substeps):
            # Share troop list for cheap separation pushes in movement
            alive_troops = [t for t in self.troops if t.hp > 0]
            for t in alive_troops:
                t._env_cache = alive_troops
            for troop in list(self.troops):
                prev_x = troop.x
                troop.update(dt, self)
                self._block_river_crossing(troop, prev_x)
            # Resolve any overlaps after movement/river blocking
            self._separate_overlaps()
            # Towers attack during each substep to keep in sync with troops
            if self.enemy_tower:
                self.enemy_tower.update(dt, self)
            if self.player_tower:
                self.player_tower.update(dt, self)
            # Let troops attack towers directly when in range
            self._troop_vs_tower_attacks()

        # Tower interactions and cleanup
        reward += self._resolve_tower_damage(prev_enemy_hp, prev_player_hp)
        reward += self._cleanup_dead_troops()

        # Termination checks
        if self.enemy_tower and self.enemy_tower.hp <= 0:
            done = True
            reward += 20.0
        if self.player_tower and self.player_tower.hp <= 0:
            done = True
            reward -= 20.0
        if self.step_count >= self.max_steps:
            done = True

        observation = self._update_grid()
        info = {
            "enemy_tower_hp": self.enemy_tower.hp if self.enemy_tower else 0,
            "player_tower_hp": self.player_tower.hp if self.player_tower else 0,
            "troop_count": len(self.troops),
        }
        return observation, reward, done, False, info

    def get_navigation_target(self, troop: Troop, target: Troop | None):
        """Return a waypoint for the troop to move toward, respecting bridges."""
        if target is None:
            # Default toward opposing tower
            target_pos = (
                0.0 if troop.friendly else float(self.rows - 1),
                float(self.cols // 2),
            )
        else:
            target_pos = (target.x, target.y)

        target_side = target_pos[0] - self.river_row
        troop_side = troop.x - self.river_row

        # If target across the river, route toward nearest bridge column first
        if target_side * troop_side <= 0:
            bridge_min = min(self.bridge_cols)
            bridge_max = max(self.bridge_cols)
            bridge_col = min(self.bridge_cols, key=lambda bc: abs(bc - troop.y))
            # Choose the nearest bridge; if equidistant between two lanes, break the tie by id
            bridge_lanes = sorted(self.bridge_cols)
            lane_choice = bridge_col
            if len(bridge_lanes) > 1 and abs(bridge_lanes[0] - bridge_lanes[1]) <= 1:
                dist_left = abs(troop.y - bridge_lanes[0])
                dist_right = abs(troop.y - bridge_lanes[1])
                if abs(dist_left - dist_right) < 1e-6:
                    lane_choice = bridge_lanes[troop.id % 2]
            if abs(troop.y - lane_choice) > self.bridge_tol * 0.5:
                # Move diagonally toward the bridge entry on current side
                approach_row = (
                    self.river_row + self.river_block_band
                    if troop_side > 0
                    else self.river_row - self.river_block_band
                )
                return (approach_row, float(lane_choice))
            # Once aligned, keep current lane within the bridge span to avoid pinning
            lane_y = troop.y
            if lane_y < bridge_min - self.bridge_tol or lane_y > bridge_max + self.bridge_tol:
                lane_y = lane_choice
            lane_y = float(np.clip(lane_y, bridge_min, bridge_max))
            return (target_pos[0], lane_y)
        return target_pos

    def get_enemy_towers(self, owner: str):
        if owner == "player":
            return [t for t in [self.enemy_tower] if t]
        return [t for t in [self.player_tower] if t]

    def _handle_player_action(self, action: int):
        if action == 0:
            return
        action -= 1
        row = action // self.cols
        col = action % self.cols

        # Only allow spawning on the player's side (bottom three rows) and empty cell
        if row < self.rows - 3:
            return
        if row == self.river_row:
            return
        if self.grid[row, col] != 0:
            return

        self._spawn_troop(
            friendly=True,
            row=row,
            col=col,
        )

    def _maybe_spawn_enemy(self):
        if self.step_count % self.enemy_spawn_every != 0:
            return
        col = int(self.rng.integers(0, self.cols))
        row = int(self.rng.integers(0, 2))  # spawn near the top
        if row == self.river_row:
            return
        if self.grid[row, col] != 0:
            return
        self._spawn_troop(friendly=False, row=row, col=col)

    def _spawn_troop(self, friendly: bool, row: int, col: int, troop_cls=None):
        """Spawn a troop of the given class (defaults to random)."""
        if troop_cls is None:
            troop_cls = [Troop, Archer][int(self.rng.integers(0, 2))]

        if troop_cls is Archer:
            troop = [Archer(id=self._next_id, friendly=friendly, x=float(row), y=float(col - 0.7)), 
                     Archer(id=self._next_id, friendly=friendly, x=float(row), y=float(col + 0.7))]
        elif troop_cls is Skeleton:
            troop = [Skeleton(id=self._next_id, friendly=friendly, x=float(row + 1), y=float(col)),
                     Skeleton(id=self._next_id, friendly=friendly, x=float(row), y=float(col + 1)),
                     Skeleton(id=self._next_id, friendly=friendly, x=float(row), y=float(col - 1))
                     ]
        else:
            troop = Troop(
                id=self._next_id,
                friendly=friendly,
                x=float(row),
                y=float(col),
                hp=760,
                attack_speed=20,
                speed=0.07,
                attack_range=1,
                sight_range=11,
                attack_damage = 86,
            )

        self._next_id += 1
        if type(troop) is list:
            for t in troop:
                self.troops.append(t)
        else:
            self.troops.append(troop)

    def _cleanup_dead_troops(self) -> float:
        reward = 0.0
        alive = []
        for troop in self.troops:
            if troop.hp <= 0:
                reward += 1.0 if troop.owner == "enemy" else -1.0
            else:
                alive.append(troop)
        self.troops = alive
        return reward

    def _resolve_tower_damage(self, prev_enemy_hp: float, prev_player_hp: float) -> float:
        reward = 0.0
        # Contact damage when troops reach tower row (legacy Clash-like behavior)
        for troop in self.troops:
            if troop.hp <= 0:
                continue
            # Friendly troops reaching enemy tower
            if troop.friendly and troop.x <= 0.0 and self.enemy_tower and self.enemy_tower.hp > 0:
                self.enemy_tower.hp -= troop.attack_damage
                troop.hp = 0
            # Enemy troops reaching player tower
            if (not troop.friendly) and troop.x >= self.rows - 1 and self.player_tower and self.player_tower.hp > 0:
                self.player_tower.hp -= troop.attack_damage
                troop.hp = 0

        if self.enemy_tower:
            reward += max(0.0, prev_enemy_hp - self.enemy_tower.hp) * 0.1
        if self.player_tower:
            reward -= max(0.0, prev_player_hp - self.player_tower.hp) * 0.1
        return reward

    def _troop_vs_tower_attacks(self):
        for troop in self.troops:
            if troop.hp <= 0:
                continue
            for tower in self.get_enemy_towers(troop.owner):
                if not tower or tower.hp <= 0:
                    continue
                if troop.distance_to(tower) <= troop.attack_range:
                    if troop.time_since_attack >= troop.attack_speed:
                        troop.time_since_attack = 0.0
                        tower.hp -= troop.attack_damage

    def _block_river_crossing(self, troop: Troop, prev_x: float):
        # Block river except at bridges; prevent lingering on river row
        delta_prev = prev_x - self.river_row
        delta_curr = troop.x - self.river_row
        col = troop.y
        near_bridge = any(abs(col - bc) <= self.bridge_tol for bc in self.bridge_cols)
        in_river_band = abs(delta_curr) <= self.river_block_band
        bridge_min = min(self.bridge_cols)
        bridge_max = max(self.bridge_cols)

        # If sitting on river without being near a bridge, push back to own side
        if in_river_band and not near_bridge:
            if delta_prev >= 0:
                troop.x = self.river_row + self.river_block_band
            else:
                troop.x = self.river_row - self.river_block_band
            return

        crosses = delta_prev * delta_curr <= 0  # includes landing exactly on river row
        if not crosses:
            return

        if near_bridge:
            # If already within the bridge span, keep current lane to let separation work
            if bridge_min - self.bridge_tol <= col <= bridge_max + self.bridge_tol:
                troop.y = float(np.clip(col, bridge_min, bridge_max))
            else:
                nearest = min(self.bridge_cols, key=lambda bc: abs(bc - col))
                troop.y = float(nearest)
            # Place the troop just over the river on the opposite side
            if delta_prev > 0:
                troop.x = self.river_row - self.river_block_band
            else:
                troop.x = self.river_row + self.river_block_band
            return

        # Hard block away from bridges
        if delta_prev >= 0:
            troop.x = max(prev_x, self.river_row + self.river_block_band)
        else:
            troop.x = min(prev_x, self.river_row - self.river_block_band)

    def _separate_overlaps(self):
        troops = [t for t in self.troops if t.hp > 0]
        n = len(troops)
        if n < 2:
            return
        for i in range(n):
            ti = troops[i]
            for j in range(i + 1, n):
                tj = troops[j]
                min_dist = max(ti.size, tj.size)
                dx = ti.x - tj.x
                dy = ti.y - tj.y
                dist2 = dx * dx + dy * dy
                if dist2 == 0.0:
                    dx, dy = 1e-3, -1e-3
                    dist2 = dx * dx + dy * dy
                if dist2 >= min_dist * min_dist:
                    continue
                dist = dist2 ** 0.5
                overlap = min_dist - dist
                if overlap <= 0.0:
                    continue
                push = (overlap * 0.5) / max(dist, 1e-6)
                offset_x = dx * push
                offset_y = dy * push
                forward_nudge = 0.0

                # Bias same-side units to push sideways rather than blocking forward
                if ti.owner == tj.owner:
                    # Relax damping at the bridge so forward movement can resolve jams
                    near_bridge_col = min(
                        abs((ti.y + tj.y) * 0.5 - bc) for bc in self.bridge_cols
                    )
                    near_bridge_lane = near_bridge_col <= self.bridge_tol + 0.1
                    near_river = max(
                        abs(ti.x - self.river_row), abs(tj.x - self.river_row)
                    ) <= self.river_block_band + 0.2
                    if near_bridge_lane and near_river:
                        # Stop backward pushes; favor side-by-side lanes
                        offset_x = 0.0
                        if abs(dy) < min_dist * 0.4:
                            side_dir = 1.0 if ti.id < tj.id else -1.0
                            lateral = max(overlap * 0.8, 0.2)
                            offset_y = lateral * side_dir
                        else:
                            offset_y *= 1.2
                        forward_nudge = 0.05  # small shared forward bump
                    else:
                        offset_x *= 0.15
                        if abs(dy) < 1e-6:
                            side_dir = 1.0 if ti.id < tj.id else -1.0
                            offset_y = (overlap * 0.5) * side_dir
                        else:
                            offset_y *= 1.1

                ti.x += offset_x
                ti.y += offset_y
                tj.x -= offset_x
                tj.y -= offset_y
                if forward_nudge:
                    ti.x += -forward_nudge if ti.friendly else forward_nudge
                    tj.x += -forward_nudge if tj.friendly else forward_nudge
                # Keep within arena bounds
                ti.x = float(np.clip(ti.x, 0.0, self.rows - 1))
                ti.y = float(np.clip(ti.y, 0.0, self.cols - 1))
                tj.x = float(np.clip(tj.x, 0.0, self.rows - 1))
                tj.y = float(np.clip(tj.y, 0.0, self.cols - 1))

    def _place_towers(self):
        if self.enemy_tower and self.enemy_tower.hp > 0:
            r = int(np.clip(round(self.enemy_tower.x), 0, self.rows - 1))
            c = int(np.clip(round(self.enemy_tower.y), 0, self.cols - 1))
            self.grid[r, c] = -2
        if self.player_tower and self.player_tower.hp > 0:
            r = int(np.clip(round(self.player_tower.x), 0, self.rows - 1))
            c = int(np.clip(round(self.player_tower.y), 0, self.cols - 1))
            self.grid[r, c] = 2

    def _update_grid(self):
        self.grid[:] = 0
        self._place_towers()
        for troop in self.troops:
            if troop.hp <= 0:
                continue
            r = int(np.clip(round(troop.x), 0, self.rows - 1))
            c = int(np.clip(round(troop.y), 0, self.cols - 1))
            self.grid[r, c] = 1 if troop.friendly else -1
        return self.grid.copy()

    def _draw_paths(self):
        # Vertical center path
        center_col = self.cols // 2
        # Split center path so it does not form a middle bridge across the river
        pygame.draw.rect(
            self.screen,
            self.path_color,
            pygame.Rect(
                center_col * self.tile_size,
                0,
                self.tile_size,
                self.river_row * self.tile_size,
            ),
        )
        pygame.draw.rect(
            self.screen,
            self.path_color,
            pygame.Rect(
                center_col * self.tile_size,
                (self.river_row + 1) * self.tile_size,
                self.tile_size,
                self.height - (self.river_row + 1) * self.tile_size,
            ),
        )

        # Side lanes to bridges
        for bridge_col in self.bridge_cols:
            pygame.draw.rect(
                self.screen,
                self.path_color,
                pygame.Rect(
                    bridge_col * self.tile_size,
                    0,
                    self.tile_size,
                    self.height,
                ),
            )

        # Cross path segments around towers (rough layout)
        padding = self.tile_size * 1.5
        thickness = self.tile_size * 1.2
        # Bottom king tower area
        pygame.draw.rect(
            self.screen,
            self.path_color,
            pygame.Rect(
                (center_col - 1) * self.tile_size,
                self.height - 3 * self.tile_size - padding,
                3 * self.tile_size,
                thickness,
            ),
        )
        # Top king tower area
        pygame.draw.rect(
            self.screen,
            self.path_color,
            pygame.Rect(
                (center_col - 1) * self.tile_size,
                padding,
                3 * self.tile_size,
                thickness,
            ),
        )

    def _draw_towers(self):
        king_positions = np.argwhere(self.grid == 2).tolist() + np.argwhere(self.grid == -2).tolist()
        for r, c in king_positions:
            is_friendly = self.grid[r, c] > 0
            color = self.tower_friendly_color if is_friendly else self.tower_enemy_color
            size = self.king_size_tiles * self.tile_size
            center_x = c * self.tile_size + self.tile_size * 0.5
            center_y = r * self.tile_size + self.tile_size * 0.5
            rect = pygame.Rect(0, 0, size, size)
            rect.center = (center_x, center_y)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (20, 20, 20), rect, 2)

            # HP bar above the tower
            tower_obj = self.player_tower if is_friendly else self.enemy_tower
            if tower_obj and tower_obj.max_hp > 0:
                hp_ratio = max(0.0, min(1.0, tower_obj.hp / tower_obj.max_hp))
                bar_width = size * 0.8
                bar_height = 8
                bar_x = center_x - bar_width / 2
                # Clamp bar_y so the enemy tower (row 0) bar stays on-screen
                bar_y = max(2, rect.top - bar_height - 6)
                pygame.draw.rect(
                    self.screen,
                    (40, 40, 40),
                    pygame.Rect(bar_x, bar_y, bar_width, bar_height),
                )
                pygame.draw.rect(
                    self.screen,
                    (50, 200, 90),
                    pygame.Rect(bar_x, bar_y, bar_width * hp_ratio, bar_height),
                )
                pygame.draw.rect(
                    self.screen,
                    (230, 230, 230),
                    pygame.Rect(bar_x, bar_y, bar_width, bar_height),
                    1,
                )

    def render(self):
        if not self.initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Clash Royale Arena")
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.SysFont("arial", 16)
            self.initialized = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                row = int(my // self.tile_size)
                col = int(mx // self.tile_size)
                # Only allow spawns on player's side (below the river) and on empty cells
                if (
                    0 <= row < self.rows
                    and 0 <= col < self.cols
                    and row > self.river_row
                    and self.grid[row, col] == 0
                ):
                    self._spawn_random_friendly(row=row, col=col)
                    self._update_grid()

        for r in range(self.grid.shape[0]):
            for c in range(self.grid.shape[1]):
                v = self.grid[r, c]

                # Base terrain checkerboard
                base_color = self.grass_light if (r + c) % 2 == 0 else self.grass_dark
                if r == self.river_row and c in self.bridge_cols:
                    base_color = (170, 130, 80)  # bridge deck
                elif r == self.river_row:
                    base_color = (70, 160, 255)  # river water

                pygame.draw.rect(
                    self.screen,
                    base_color,
                    pygame.Rect(
                        c * self.tile_size,
                        r * self.tile_size,
                        self.tile_size,
                        self.tile_size,
                    ),
        )

        # Draw paths/lanes
        self._draw_paths()

        # Draw towers on top of paths/ground
        self._draw_towers()

        # Draw troops at continuous positions (not snapped to grid)
        for troop in self.troops:
            if troop.hp <= 0:
                continue
            color = (50, 150, 255) if troop.friendly else (255, 80, 50)
            troop_color = color_map[troop.name]
            px = troop.y * self.tile_size + self.tile_size * 0.5
            py = troop.x * self.tile_size + self.tile_size * 0.5
            pygame.draw.circle(
                self.screen,
                color,
                (int(px), int(py)),
                int(self.tile_size * 0.5),
            )
            
            pygame.draw.circle(
                self.screen,
                troop_color,
                (int(px), int(py)),
                int(self.tile_size * 0.3),
            )
            
            # HP bar above the troop
            if troop.max_hp > 0:
                hp_ratio = max(0.0, min(1.0, troop.hp / troop.max_hp))
                bar_width = self.tile_size * 1.5
                bar_height = 5
                bar_x = px - bar_width / 2
                bar_y = py - self.tile_size * 0.6 - bar_height
                pygame.draw.rect(
                    self.screen,
                    (40, 40, 40),
                    pygame.Rect(bar_x, bar_y, bar_width, bar_height),
                )
                pygame.draw.rect(
                    self.screen,
                    (50, 200, 90),
                    pygame.Rect(bar_x, bar_y, bar_width * hp_ratio, bar_height),
                )
                pygame.draw.rect(
                    self.screen,
                    (230, 230, 230),
                    pygame.Rect(bar_x, bar_y, bar_width, bar_height),
                    1,
                )

        # HUD overlay
        friendly_count = sum(1 for t in self.troops if t.friendly and t.hp > 0)
        enemy_count = sum(1 for t in self.troops if (not t.friendly) and t.hp > 0)
        hud_lines = [
            f"Step: {self.step_count}/{self.max_steps}",
            f"Player tower HP: {self.player_tower.hp if self.player_tower else 0}",
            f"Enemy tower HP: {self.enemy_tower.hp if self.enemy_tower else 0}",
            f"Friendly troops: {friendly_count}",
            f"Enemy troops: {enemy_count}",
        ]
        hud_bg = pygame.Rect(5, 5, 200, 100)
        pygame.draw.rect(self.screen, (30, 30, 30), hud_bg)
        pygame.draw.rect(self.screen, (200, 200, 200), hud_bg, 1)
        for i, line in enumerate(hud_lines):
            surf = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(surf, (10, 10 + i * 18))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.initialized:
            pygame.quit()
            self.initialized = False


if __name__ == "__main__":
    env = ClashRoyaleEnv()
    obs, _ = env.reset()
    done = False
    while not done:
        # Manual play: left-click to place troops on your side of the arena
        action = 0
        obs, reward, done, _, info = env.step(action)
        env.render()
    env.close()
