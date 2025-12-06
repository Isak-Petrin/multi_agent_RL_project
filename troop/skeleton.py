from troop.troop import Troop


class Skeleton(Troop):
    DEFAULT_COST = 2

    def __init__(self, id, friendly, x, y, cost=None):
        super().__init__(
            id = id,
            friendly= friendly,
            x = x,
            y = y,
            hp = 32,
            attack_speed = 20,
            speed = 1.1,
            attack_range = 1.1,
            sight_range = 5.5,
            targets="ground",
            size=0.6,
            attack_damage = 35,
            cost=cost,
        )
            
