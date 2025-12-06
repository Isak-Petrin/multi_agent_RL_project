from troop.troop import Troop


class Giant(Troop):
    DEFAULT_COST = 5

    def __init__(self, id, friendly, x, y, cost=None):
        super().__init__(
            id = id,
            friendly= friendly,
            x = x,
            y = y,
            hp = 2125,
            attack_speed = 1.5,
            speed = 0.9,
            attack_range = 1.2,
            sight_range = 5.5,
            targets="ground",
            target_type="towers",
            size=0.8,
            attack_damage = 131,
            cost=cost,
        )
            
