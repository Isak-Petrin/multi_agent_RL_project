from troop.troop import Troop


class Archer(Troop):
    DEFAULT_COST = 3

    def __init__(self, id, friendly, x, y, cost=None):
        super().__init__(
            id = id,
            friendly= friendly,
            x = x,
            y = y,
            hp = 130,
            attack_speed = 0.9,
            speed = 1.2,
            attack_range = 5,
            sight_range = 5.5,
            targets="ground",
            size=0.6,
            attack_damage = 48,
            cost=cost,
        )
            
