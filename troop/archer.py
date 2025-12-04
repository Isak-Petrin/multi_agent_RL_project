from troop.troop import Troop
class Archer(Troop):
    def __init__(self, id, friendly, x, y):
        super().__init__(
            id = id,
            friendly= friendly,
            x = x,
            y = y,
            hp = 300,
            attack_speed = 20,
            speed = 0.07,
            attack_range = 5,
            sight_range = 5.5,
            targets="ground",
            size=0.6,
            attack_damage = 44,
        )
            