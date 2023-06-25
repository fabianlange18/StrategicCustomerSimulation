import config

class Undercutting_Vendor():

    def __init__(self):
        self.price = config.reference_price
        self.price_ceiling = config.undercutting_competitor_ceiling
        self.undercut_amount = config.undercutting_competitor_step
        self.price_floor = config.undercutting_competitor_floor
    
    def update_price(self, market_price):
        self.price = market_price - self.undercut_amount
        if self.price < self.price_floor:
            self.price = self.price_ceiling if self.price_ceiling else self.price_floor
        return self.price