class Troop:
    DEFAULT_COST = 4

    def __init__(
        self,
        id: int,
        friendly: bool,
        x: float,
        y: float,
        hp: int,
        attack_speed: float,
        speed: float,
        attack_range: float,
        sight_range: float,
        targets="ground",
        target_type: str = "all",
        size=0.6,
        attack_damage: int = 10,
        cost: int | None = None,
        name: str | None = None,
    ):
        valid_target_types = {"troops", "towers", "all"}
        if target_type not in valid_target_types:
            raise ValueError(
                f"Invalid target_type '{target_type}'. Expected one of {valid_target_types}."
            )
        self.id = id
        self.name = name or type(self).__name__
        self.is_troop = True
        # True if troop is owned by player and False if troop is owned by opposition
        self.friendly = friendly
        self.owner = "player" if friendly else "enemy"

        # Position of troop (grid coordinates)
        self.x = x
        self.y = y

        self.hp = hp
        self.attack_speed = attack_speed  # seconds between attacks
        self.speed = speed  # tiles per second
        self.attack_range = attack_range
        self.targets = targets
        self.sight_range = sight_range
        self.size = size
        self.attack_damage = attack_damage
        self.cost = self.DEFAULT_COST if cost is None else cost
        self.target_type = target_type
        self.targets_towers = target_type in {"towers", "all"}
        self.targets_troops = target_type in {"troops", "all"}

        self.max_hp = hp
        self.time_since_attack = 0.0
        self.target = None

    def distance_to(self, enemy) -> float:
        return ((self.x - enemy.x) ** 2 + (self.y - enemy.y) ** 2) ** 0.5

    def update(self, dt: float, env):
        self.time_since_attack += dt

        if self.target is None or self._target_invalid():
            self.target = self._acquire_target(env)

        nav_target = env.get_navigation_target(self, self.target)

        if self.target and self.distance_to(self.target) <= self.attack_range:
            self._attack_target_if_ready()
        else:
            self._move_towards_target(dt, nav_target)

    def _acquire_target(self, env):
        candidates = []
        if self.targets_troops:
            candidates.extend(
                t
                for t in env.troops
                if t.owner != self.owner and t.hp > 0 and self._can_target(t)
            )

        if self.targets_towers:
            for tower in env.get_enemy_towers(self.owner):
                if tower.hp > 0 and self._can_target(tower):
                    candidates.append(tower)

        if not candidates:
            return None

        # Filter only enemies within sight range
        visible_enemies = [
            e for e in candidates if self.distance_to(e) <= self.sight_range
        ]

        if not visible_enemies:
            return None

        return min(visible_enemies, key=lambda e: self.distance_to(e))

    def _target_invalid(self):
        if self.target is None:
            return True
        if self.target.hp <= 0:
            return True
        if self.distance_to(self.target) > self.sight_range:
            return True
        if getattr(self.target, "is_building", False) and not self.targets_towers:
            return True
        if getattr(self.target, "is_troop", False) and not self.targets_troops:
            return True
        return False

    def _can_target(self, target) -> bool:
        if self.targets == "all":
            return True
        if self.targets == "ground" and getattr(target, "is_air", False):
            return False
        return True

    def _move_towards_target(self, dt: float, nav_target):
        dx = nav_target[0] - self.x
        dy = nav_target[1] - self.y

        # Soft separation to avoid stacking; cheap because troop counts are small
        sep_x = sep_y = 0.0
        for other in getattr(self, "_env_cache", []) or []:
            if other is self or other.hp <= 0:
                continue
            # Only nudge away if truly overlapping; use larger radius as spacing
            cushion = max(self.size, getattr(other, "size", self.size))
            cushion2 = cushion * cushion
            ox = self.x - other.x
            oy = self.y - other.y
            dist2 = ox * ox + oy * oy
            if dist2 == 0.0:
                # Identical position: pick a tiny deterministic offset to separate
                ox, oy = 1e-3, -1e-3
                dist2 = ox * ox + oy * oy
            if dist2 >= cushion2:
                continue
            dist = dist2 ** 0.5
            inv_dist = 1.0 / max(dist, 1e-6)
            penetration = cushion - dist
            push = penetration * 0.8  # push hard enough to clear overlap quickly
            sep_x += ox * inv_dist * push
            sep_y += oy * inv_dist * push

        # Don't let separation push against forward motion; keep sideways bias
        forward_dir = -1.0 if self.friendly else 1.0
        if sep_x * forward_dir < 0.0:
            sep_x *= 0.2  # damp backward push

        dx += sep_x
        dy += sep_y

        dist = max((dx**2 + dy**2) ** 0.5, 1e-6)
        step = min(self.speed * dt, dist)
        self.x += (dx / dist) * step
        self.y += (dy / dist) * step


    def _attack_target_if_ready(self):
        if self.time_since_attack < self.attack_speed:
            return
        self.time_since_attack = 0.0
        self.target.hp -= self.attack_damage
