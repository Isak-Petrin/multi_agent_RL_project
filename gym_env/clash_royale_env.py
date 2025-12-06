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
from troop.giant import Giant
from troop.tower import Tower
from troop.troop import Troop
from troop.skeleton import Skeleton

color_map = {"Archer": (179, 0, 255),
             "Skeleton": (255,255,255),
             "Troop": (0,0,0),
             "Giant": (255, 180, 0)}

class ClashRoyaleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, max_steps: int = 10000, enemy_spawn_every: int = 180):
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
        self.arena_height = self.rows * self.tile_size
        self.ui_panel_height = 110
        self.height = self.arena_height + self.ui_panel_height
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
        self.sim_substeps = 20  # smaller dt increments per env step to smooth motion
        self.player_hand = [Troop, Archer, Skeleton, Giant]
        self.selected_hand_index = 0
        self.max_elixir = 10.0
        self.base_elixir_rate = 1.0 / 2.8
        self.double_elixir_rate = 1.0 / 1.4
        self.double_elixir_time = 60.0  # seconds
        self.player_elixir = 5.0
        self.enemy_elixir = 5.0
        self.elapsed_time = 0.0

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
        self.player_elixir = 5.0
        self.enemy_elixir = 5.0
        self.elapsed_time = 0.0
        self.selected_hand_index = 0
        self._place_towers()
        observation = self._update_grid()
        return observation, {}

    def _spawn_random_friendly(self, row: int, col: int):
        """Spawn a random friendly troop at the specified cell."""
        troop_choices = [Troop, Archer, Skeleton, Giant]  # Extend here with more troop classes
        idx = int(self.rng.integers(0, len(troop_choices)))
        troop_cls = troop_choices[idx]
        return self._spawn_troop(friendly=True, row=row, col=col, troop_cls=troop_cls)

    def _spawn_selected_friendly(self, row: int, col: int):
        """Spawn the currently selected card from the player's hand."""
        if not self.player_hand:
            return False
        troop_cls = self.player_hand[self.selected_hand_index % len(self.player_hand)]
        return self._spawn_troop(friendly=True, row=row, col=col, troop_cls=troop_cls)

    def _init_towers(self):
        center_col = self.cols // 2
        self.enemy_tower = Tower(
            friendly=False,
            x=0.0,
            y=float(center_col),
            hp=2500,
            attack_speed=20,  # slower cadence so troops can approach
            attack_range=7,  # outranged by archers to make chip damage possible
            attack_damage=58,
        )
        self.player_tower = Tower(
            friendly=True,
            x=float(self.rows - 1),
            y=float(center_col),
            hp=2500,
            attack_speed=20,
            attack_range=7,
            attack_damage=58,
        )

    def step(self, action):
        self.step_count += 1
        
        # 1. Base Time Penalty (Encourage winning fast)
        reward = -0.01 
        done = False

        # 2. Handle Action & Apply Invalid Action Penalty
        # (Change _handle_player_action to return the penalty calculated in A)
        invalid_action_penalty = self._handle_player_action(action)
        reward += invalid_action_penalty

        self._maybe_spawn_enemy()

        prev_enemy_hp = self.enemy_tower.hp if self.enemy_tower else 0
        prev_player_hp = self.player_tower.hp if self.player_tower else 0

        # --- Physics Loop (Unchanged) ---
        dt = 1.0 / self.sim_substeps
        for _ in range(self.sim_substeps):
            self._update_elixir(dt)
            alive_troops = [t for t in self.troops if t.hp > 0]
            for t in alive_troops:
                t._env_cache = alive_troops
            for troop in list(self.troops):
                prev_x = troop.x
                troop.update(dt, self)
                self._block_river_crossing(troop, prev_x)
            self._separate_overlaps()
            if self.enemy_tower:
                self.enemy_tower.update(dt, self)
            if self.player_tower:
                self.player_tower.update(dt, self)
            self._troop_vs_tower_attacks()
        # --------------------------------

        # 3. Resolve Tower Damage (Dense Reward)
        # Using 0.1 scale
        tower_reward = self._resolve_tower_damage(prev_enemy_hp, prev_player_hp)
        reward += tower_reward

        # 4. Resolve Elixir Trades (Strategic Reward)
        # This replaces the simple +/- 1.0 logic
        elixir_reward = self._cleanup_dead_troops()
        reward += elixir_reward

        # 5. Winning / Losing (Terminal Reward)
        # Increase the magnitude to overpower the accumulated step penalties
        if self.enemy_tower and self.enemy_tower.hp <= 0:
            done = True
            reward += 100.0 # Big bonus for the kill
        if self.player_tower and self.player_tower.hp <= 0:
            done = True
            reward -= 100.0
        if self.step_count >= self.max_steps:
            done = True
            # Optional: Tie-breaker reward based on HP difference?
            # hp_diff = (self.player_tower.hp - self.enemy_tower.hp) / 100.0
            # reward += hp_diff

        observation = self._update_grid()
        
        # Useful for debugging training performance in Tensorboard
        info = {
            "enemy_tower_hp": self.enemy_tower.hp if self.enemy_tower else 0,
            "player_tower_hp": self.player_tower.hp if self.player_tower else 0,
            "troop_count": len(self.troops),
            "player_elixir": self.player_elixir,
            "rewards/tower": tower_reward,
            "rewards/elixir": elixir_reward,
            "rewards/invalid": invalid_action_penalty,
            "troop_placement": self.troops
        }
        return observation, reward, done, False, info

    def step_self_play(self, bottom_action: int, top_action: int):
        """
        Advance the game with actions from both sides while keeping each agent's ego
        perspective (both think they deploy from the bottom). Dynamics remain unchanged;
        this only alters how actions/observations are mapped for the enemy side.
        """
        self.step_count += 1
        reward_bottom = -0.01
        done = False

        # Apply both agents' spawn actions (top action is flipped into world coords)
        self._handle_spawn_action(bottom_action, friendly=True)
        top_world_action = self._ego_action_to_world(top_action, friendly=False)
        self._handle_spawn_action(top_world_action, friendly=False)

        prev_enemy_hp = self.enemy_tower.hp if self.enemy_tower else 0
        prev_player_hp = self.player_tower.hp if self.player_tower else 0

        dt = 1.0 / self.sim_substeps
        for _ in range(self.sim_substeps):
            self._update_elixir(dt)
            alive_troops = [t for t in self.troops if t.hp > 0]
            for t in alive_troops:
                t._env_cache = alive_troops
            for troop in list(self.troops):
                prev_x = troop.x
                troop.update(dt, self)
                self._block_river_crossing(troop, prev_x)
            self._separate_overlaps()
            if self.enemy_tower:
                self.enemy_tower.update(dt, self)
            if self.player_tower:
                self.player_tower.update(dt, self)
            self._troop_vs_tower_attacks()

        reward_bottom += self._resolve_tower_damage(prev_enemy_hp, prev_player_hp)
        reward_bottom += self._cleanup_dead_troops()

        if self.enemy_tower and self.enemy_tower.hp <= 0:
            done = True
            reward_bottom += 20.0
        if self.player_tower and self.player_tower.hp <= 0:
            done = True
            reward_bottom -= 20.0
        if self.step_count >= self.max_steps:
            done = True

        world_obs = self._update_grid()
        obs_bottom = world_obs.copy()
        obs_top = self._ego_observation_from_grid(world_obs, friendly=False)

        reward_top = -reward_bottom  # zero-sum mirror of bottom reward
        info = {
            "enemy_tower_hp": self.enemy_tower.hp if self.enemy_tower else 0,
            "player_tower_hp": self.player_tower.hp if self.player_tower else 0,
            "troop_count": len(self.troops),
            "player_elixir": self.player_elixir,
            "enemy_elixir": self.enemy_elixir,
            "elapsed_time": self.elapsed_time,
            "bottom_action": bottom_action,
            "top_action_world": top_world_action,
        }
        return (obs_bottom, obs_top), (reward_bottom, reward_top), done, False, info

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

    def _valid_spawn_cell(self, friendly: bool, row: int, col: int) -> bool:
        """Check if a spawn cell is valid for the given side."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if row == self.river_row:
            return False
        if self.grid[row, col] != 0:
            return False
        if friendly:
            return row >= self.rows - 3
        return row < 3
    """
    def _handle_spawn_action(self, action: int, friendly: bool) -> bool:
        if action == 0:
            return False
        action -= 1
        row = action // self.cols
        col = action % self.cols
        if not self._valid_spawn_cell(friendly, row, col):
            return False
        return bool(self._spawn_troop(friendly=friendly, row=row, col=col))
    
    # In your Class...
    """

    def _handle_spawn_action(self, action: int, friendly: bool) -> float:
        """
        Handle a discrete spawn action.
        Returns: A small negative reward if the move was invalid (e.g. not enough elixir), else 0.0
        """
        if action == 0:
            return 0.0 # No-op is always valid and costs nothing
        
        # Parse action
        action_idx = action - 1
        row = action_idx // self.cols
        col = action_idx % self.cols
        
        # 1. Check bounds/validity
        if not self._valid_spawn_cell(friendly, row, col):
            return -0.1  # Penalty for trying to spawn in river or on top of another unit
            
        # 2. Determine troop type (Logic from your original code needs to be accessible here)
        # Assuming you determine class randomly or via action, let's look at how you did it in _spawn_troop.
        # NOTE: To strictly calculate cost before spawning, we need the intended class.
        # For this example, let's assume the agent selects the card type in the action 
        # (if your action space separates Card/Position). 
        # If your current code spawns a RANDOM troop, the agent can't predict cost. 
        # Assuming for now we just penalize the spawn attempt if it failed due to funds:
        
        spawned = self._spawn_troop(friendly=friendly, row=row, col=col)
        if not spawned:
            # If _spawn_troop returned False, it was likely due to insufficient elixir
            return -0.05 
            
        return 0.0

    def _current_elixir_rate(self) -> float:
        if self.elapsed_time >= self.double_elixir_time:
            return self.double_elixir_rate
        return self.base_elixir_rate

    def _format_time(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _update_elixir(self, dt: float):
        rate = self._current_elixir_rate()
        self.player_elixir = min(self.max_elixir, self.player_elixir + rate * dt)
        self.enemy_elixir = min(self.max_elixir, self.enemy_elixir + rate * dt)
        self.elapsed_time += dt

    def _get_troop_cost(self, troop_cls) -> float:
        if troop_cls is None:
            troop_cls = Troop
        return float(getattr(troop_cls, "DEFAULT_COST", 1))

    def _try_pay_elixir(self, friendly: bool, cost: float) -> bool:
        if friendly:
            if self.player_elixir < cost:
                return False
            self.player_elixir -= cost
            return True
        if self.enemy_elixir < cost:
            return False
        self.enemy_elixir -= cost
        return True

    def _ego_action_to_world(self, action: int, friendly: bool) -> int:
        """Convert an ego-centric action to world coordinates (vertical flip for enemy)."""
        if friendly or action == 0:
            return action
        action -= 1
        row = action // self.cols
        col = action % self.cols
        world_row = (self.rows - 1) - row
        world_col = col
        return world_row * self.cols + world_col + 1

    def _ego_observation_from_grid(self, grid: np.ndarray, friendly: bool) -> np.ndarray:
        """Return a perspective-correct grid for the given side (enemy sees itself as positive)."""
        if friendly:
            return grid.copy()
        flipped = np.flipud(grid)
        return -flipped

    def get_ego_observation(self, friendly: bool) -> np.ndarray:
        """Public helper to obtain an ego-centric observation without altering state."""
        grid = self._update_grid()
        return self._ego_observation_from_grid(grid, friendly)

    def _handle_player_action(self, action: int) -> float:
        """Process the player's action and return any invalid-action penalty."""
        penalty = self._handle_spawn_action(action, friendly=True)
        # _handle_spawn_action should always return a float, but guard against None to avoid TypeErrors.
        return 0.0 if penalty is None else float(penalty)

    def _maybe_spawn_enemy(self):
        if self.step_count % self.enemy_spawn_every != 0:
            return
        col = int(self.rng.integers(0, self.cols))
        row = int(self.rng.integers(0, 2))  # spawn near the top
        if not self._valid_spawn_cell(friendly=False, row=row, col=col):
            return
        self._spawn_troop(friendly=False, row=row, col=col)

    def _spawn_troop(self, friendly: bool, row: int, col: int, troop_cls=None):
        """Spawn a troop of the given class (defaults to random)."""
        if troop_cls is None:
            troop_pool = [Troop, Archer, Skeleton, Giant]
            troop_cls = troop_pool[int(self.rng.integers(0, len(troop_pool)))]

        cost = self._get_troop_cost(troop_cls)
        if not self._try_pay_elixir(friendly, cost):
            return False

        if troop_cls is Archer:
            troop = [Archer(id=self._next_id, friendly=friendly, x=float(row), y=float(col - 0.7)), 
                     Archer(id=self._next_id, friendly=friendly, x=float(row), y=float(col + 0.7))]
        elif troop_cls is Skeleton:
            troop = [Skeleton(id=self._next_id, friendly=friendly, x=float(row + 1), y=float(col)),
                     Skeleton(id=self._next_id, friendly=friendly, x=float(row), y=float(col + 1)),
                     Skeleton(id=self._next_id, friendly=friendly, x=float(row), y=float(col - 1))
                     ]
        elif troop_cls is Giant:
            troop = Giant(
                id=self._next_id,
                friendly=friendly,
                x=float(row),
                y=float(col),
            )
        else:
            troop = Troop(
                id=self._next_id,
                friendly=friendly,
                x=float(row),
                y=float(col),
                hp=288,
                attack_speed=1.3,
                speed=1.2,
                attack_range=1,
                sight_range=5.5,
                attack_damage = 82,
            )

        self._next_id += 1
        if type(troop) is list:
            for t in troop:
                self.troops.append(t)
        else:
            self.troops.append(troop)
        return True

    def _cleanup_dead_troops(self) -> float:
            reward = 0.0
            alive = []
            for troop in self.troops:
                if troop.hp <= 0:
                    # Calculate Elixir Value of the dead unit
                    # You need to ensure every troop instance has a .cost attribute or map it here
                    unit_cost = self._get_troop_cost(type(troop))
                    
                    if troop.owner == "enemy":
                        # We killed an enemy! Reward = value of that enemy
                        reward += unit_cost * 1.0 
                    else:
                        # We lost a unit! Penalty = value of that unit
                        reward -= unit_cost * 1.0
                else:
                    alive.append(troop)
            self.troops = alive
            return reward

    def _resolve_tower_damage(self, prev_enemy_hp: float, prev_player_hp: float) -> float:
        reward = 0.0
        # Contact damage when troops reach tower row (legacy Clash-like behavior)
        for troop in self.troops:
            if troop.hp <= 0 or not getattr(troop, "targets_towers", False):
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
            if troop.hp <= 0 or not getattr(troop, "targets_towers", False):
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
                self.arena_height - (self.river_row + 1) * self.tile_size,
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
                    self.arena_height,
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
                self.arena_height - 3 * self.tile_size - padding,
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

    def _draw_elixir_ui(self):
        panel_top = self.arena_height
        panel_rect = pygame.Rect(0, panel_top, self.width, self.ui_panel_height)
        pygame.draw.rect(self.screen, (60, 45, 80), panel_rect)
        pygame.draw.rect(self.screen, (110, 90, 140), panel_rect, 2)

        card_w = 80
        card_h = self.ui_panel_height - 40
        card_pad = 14
        card_y = panel_top + 8
        for i, troop_cls in enumerate(self.player_hand):
            x = card_pad + i * (card_w + card_pad)
            card_rect = pygame.Rect(x, card_y, card_w, card_h)
            selected = i == (self.selected_hand_index % max(1, len(self.player_hand)))
            cost = self._get_troop_cost(troop_cls)
            affordable = self.player_elixir >= cost
            fill_color = (130, 110, 90) if affordable else (80, 70, 80)
            pygame.draw.rect(self.screen, fill_color, card_rect, border_radius=8)
            border_color = (230, 200, 90) if selected else (30, 30, 30)
            pygame.draw.rect(self.screen, border_color, card_rect, width=3 if selected else 2, border_radius=8)

            name_label = troop_cls.__name__
            name_surf = self.font.render(name_label, True, (240, 240, 240))
            name_rect = name_surf.get_rect(center=(card_rect.centerx, card_rect.top + 14))
            self.screen.blit(name_surf, name_rect)

            cost_radius = 14
            cost_center = (card_rect.centerx, card_rect.bottom - cost_radius - 4)
            bubble_color = (200, 70, 200) if affordable else (120, 80, 130)
            pygame.draw.circle(self.screen, bubble_color, cost_center, cost_radius)
            cost_surf = self.font.render(str(int(cost)), True, (255, 255, 255))
            cost_rect = cost_surf.get_rect(center=cost_center)
            self.screen.blit(cost_surf, cost_rect)

        bar_margin = 12
        bar_height = 18
        bar_y = panel_top + self.ui_panel_height - bar_height - 10
        bar_rect = pygame.Rect(bar_margin, bar_y, self.width - 2 * bar_margin, bar_height)
        pygame.draw.rect(self.screen, (40, 30, 60), bar_rect, border_radius=10)
        ratio = max(0.0, min(1.0, self.player_elixir / self.max_elixir))
        fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, int(bar_rect.width * ratio), bar_height)
        pygame.draw.rect(self.screen, (230, 80, 210), fill_rect, border_radius=10)
        pygame.draw.rect(self.screen, (210, 180, 230), bar_rect, 2, border_radius=10)

        elixir_text = f"Elixir: {self.player_elixir:.1f}/{int(self.max_elixir)}"
        elixir_surf = self.font.render(elixir_text, True, (245, 245, 245))
        self.screen.blit(elixir_surf, (bar_rect.x + 8, bar_rect.y - 18))

        rate = self._current_elixir_rate()
        if self.elapsed_time >= self.double_elixir_time:
            status_text = f"2x Elixir ({rate:.2f}/s)"
        else:
            eta = max(0, self.double_elixir_time - self.elapsed_time)
            status_text = f"2x in {int(eta)}s ({rate:.2f}/s)"
        status_surf = self.font.render(status_text, True, (240, 220, 250))
        status_rect = status_surf.get_rect(right=self.width - bar_margin, centery=bar_rect.y - 10)
        self.screen.blit(status_surf, status_rect)

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
            if event.type == pygame.KEYDOWN:
                if pygame.K_1 <= event.key <= pygame.K_4:
                    idx = event.key - pygame.K_1
                    if idx < len(self.player_hand):
                        self.selected_hand_index = idx
                elif event.key == pygame.K_TAB:
                    if self.player_hand:
                        self.selected_hand_index = (self.selected_hand_index + 1) % len(self.player_hand)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if my >= self.arena_height:
                    continue  # clicked on UI panel
                row = int(my // self.tile_size)
                col = int(mx // self.tile_size)
                # Only allow spawns on player's side (below the river) and on empty cells
                if (
                    0 <= row < self.rows
                    and 0 <= col < self.cols
                    and row > self.river_row
                    and self.grid[row, col] == 0
                ):
                    if self._spawn_selected_friendly(row=row, col=col):
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

        # Elixir / hand UI
        self._draw_elixir_ui()

        # HUD overlay
        friendly_count = sum(1 for t in self.troops if t.friendly and t.hp > 0)
        enemy_count = sum(1 for t in self.troops if (not t.friendly) and t.hp > 0)
        hud_lines = [
            f"Step: {self.step_count}/{self.max_steps}",
            f"Player tower HP: {self.player_tower.hp if self.player_tower else 0}",
            f"Enemy tower HP: {self.enemy_tower.hp if self.enemy_tower else 0}",
            f"Friendly troops: {friendly_count}",
            f"Enemy troops: {enemy_count}",
            f"Elixir: {self.player_elixir:.1f}/{self.max_elixir}",
            f"Time: {self._format_time(self.elapsed_time)}",
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
