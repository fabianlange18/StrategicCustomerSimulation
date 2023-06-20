import config

class Undercutting_Vendor():

    def __init__(self):
        self.price_floor = config.undercutting_competitor_floor
        self.price_ceiling = config.undercutting_competitor_ceiling
        self.price = config.reference_price
    
    def update_price(self, market_price):
        self.price = market_price - 1
        if self.price < self.price_floor:
            self.price = self.price_ceiling if self.price_ceiling else self.price_floor
        return self.price