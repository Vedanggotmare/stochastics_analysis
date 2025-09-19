import numpy as np
import random 
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

def auction_step(n , init_price = 10, init_price_std = 1, volume_buy = 20, volume_sell = 20):
    buys = [random.normalvariate(init_price,init_price_std) for _ in range(n)]
    buy = sorted([[buys[i],volume_buy] for i in range(n)], reverse= True)

    sells = [random.normalvariate(init_price,init_price_std) for _ in range(n)]
    sell = sorted([[sells[i],volume_sell] for i in range(n)], reverse= False)

    buy_prices = [b[0] for b in buy]
    buy_volumes = [b[1] for b in buy]
    buy_cumvol = np.cumsum(buy_volumes)

    sell_prices = [s[0] for s in sell]
    sell_volumes = [s[1] for s in sell]
    sell_cumvol = np.cumsum(sell_volumes)

    sort_prices = sorted(set(buy_prices + sell_prices))

    auction_price = None
    cleared_vol = -1

    for p in sort_prices:
        demand = sum(q for price, q in buy if price >= p)
        supply = sum(q for price, q in sell if price <= p)
        traded = min(demand,supply)
        if traded >= cleared_vol:
            cleared_vol = traded
            auction_price = p

    return auction_price, cleared_vol, buy_cumvol, sell_cumvol, buy_prices, sell_prices

if __name__ == '__main__':

    steps = 100000
    time_steps = range(steps)

    init_price = 10
    prices = []

    for _ in tqdm(time_steps):
        result = auction_step(
            100,
            init_price=init_price,  # use last auction price
            volume_buy=int(10*np.random.chisquare(3)), 
            volume_sell=int(10*np.random.chisquare(3))
        )
        auction_price, cleared_vol, buy_cumvol, sell_cumvol, buy_prices, sell_prices = result
        prices.append(auction_price)

        # feedback: update init_price for next round
        init_price = auction_price

    # take the last order book snapshot
    auction_price, cleared_vol, buy_cumvol, sell_cumvol, buy_prices, sell_prices = result

    # plot price evolution
    plt.figure(figsize=(10,6))
    plt.plot(time_steps, prices)
    plt.xlabel("Time step")
    plt.ylabel("Auction Price")
    plt.grid(True)
    plt.show()

    # plot final order book
    plt.step(buy_cumvol, buy_prices, color="green", where="post", label="Buy")
    plt.step(sell_cumvol, sell_prices, color="red", where="post", label="Sell")
    plt.axhline(auction_price, linestyle="--", color="black", label="Auction Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.hist(prices, bins= 80)
    plt.grid(True)
    plt.show()
