class Tower:
    def __init__(
        self,
        friendly: bool,
        x: float,
        y: float,
        hp: int,
        attack_speed: float,
        attack_range: float,
        attack_damage: int,
    ):
        self.friendly = friendly
        self.owner = "player" if friendly else "enemy"
        self.x = x
        self.y = y
        self.is_building = True
        self.hp = hp
        self.max_hp = hp
        self.attack_speed = attack_speed
        self.attack_range = attack_range
        self.attack_damage = attack_damage
        self.time_since_attack = 0.0

    def distance_to(self, target) -> float:
        return ((self.x - target.x) ** 2 + (self.y - target.y) ** 2) ** 0.5

    def _acquire_target(self, env):
        return min(
            (t for t in env.troops if t.owner != self.owner and t.hp > 0),
            default=None,
            key=lambda t: self.distance_to(t),
        )

    def _attack_target_if_ready(self, target):
        if self.time_since_attack < self.attack_speed:
            return
        self.time_since_attack = 0.0
        target.hp -= self.attack_damage

    def update(self, dt: float, env):
        if self.hp <= 0:
            return
        self.time_since_attack += dt
        target = self._acquire_target(env)
        if target and self.distance_to(target) <= self.attack_range:
            self._attack_target_if_ready(target)
