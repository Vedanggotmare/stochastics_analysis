from market import Market
import random
import matplotlib.pyplot as plt

def main():
    market = Market()
    mid_prices = []
    last_trades = []

    for t in range(10000):
        action = random.choice(["limit", "market", "cancel"])
        side = random.choice(["buy", "sell"])
        price = random.normalvariate(10,1)
        volume = random.randint(100, 500)

        if action == "limit":
            market.insert_limit_order(price, volume, side)
        elif action == "market":
            market.insert_market_order(volume, side)
        elif action == "cancel":
            market.cancel_order(price, side)

        mid_prices.append(market.mid_price())
        last_trades.append(market.last_trade)

        # print(f"Step {t}: {action} {side} {volume}@{price}")
        # print(f"  Book: {market.snapshot()}")
        # print(f"  Mid-price: {market.mid_price()} | Last trade: {market.last_trade}\n")

    # Plot evolution
    plt.plot(mid_prices, label="Mid-price")
    #plt.plot(last_trades, label="Last trade price", linestyle="--")
    plt.legend()
    plt.title("Simulated Market Price Evolution")
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.show()

if __name__ == "__main__":
    main()
