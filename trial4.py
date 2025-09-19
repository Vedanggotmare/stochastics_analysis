import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------
# Config / hyperparameters
# -------------------------
STEPS = 5000               # number of auctions (time steps)
N_ORDERS = 500             # orders per side per auction
INIT_FAIR = 10.0           # initial fundamental / fair value
INIT_MONEY = 1_000_000.0   # initial aggregate money pool (cash)
TAX_RATE = 0.001           # transaction tax fraction (0.1% per trade)
LIQ_INJECT_FREQ = 500      # every this many steps inject liquidity
LIQ_INJECT_MEAN = 50_000   # mean injection amount
LIQ_INJECT_STD = 5_000
NEWS_PROB = 0.02           # per-step probability of a news event
NEWS_SIGMA_SMALL = 0.2     # small news volatility
NEWS_BIG_PROB = 0.01       # when news happens, chance of big shock
NEWS_SIGMA_BIG = 2.0       # big news scale
FUND_DRIFT = 0.0           # drift per step for fair value (additive)
FUND_SIGMA = 0.05          # fundamental volatility per step (additive)
MIN_PRICE = 0.01           # floor for prices

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# -------------------------
# Helper: sample volume model
# -------------------------
def sample_volume():
    # mixture: 95% lognormal small orders, 5% heavy Pareto block
    if np.random.rand() < 0.95:
        v = int(np.random.lognormal(mean=3.0, sigma=0.6))  # median ~ exp(3)=20
    else:
        v = int(1 + np.random.pareto(2.0) * 200)            # occasional large block
    # clamp
    return max(1, min(v, 200_000))

# -------------------------
# Auction clearing (vectorized-ish)
# -------------------------
def find_clearing_price_and_volume(buy_prices, buy_q, sell_prices, sell_q):
    """
    buy_prices: np.array descending (high -> low)
    buy_q: np.array matching quantities
    sell_prices: np.array ascending (low -> high)
    sell_q: np.array matching quantities
    Returns: auction_price, cleared_volume
    """
    # cumulative volumes
    cum_buy = np.cumsum(buy_q)    # demand curve at descending buy_prices
    cum_sell = np.cumsum(sell_q)  # supply curve at ascending sell_prices

    # candidate price grid: unique union of buy/sell prices (sorted ascending)
    all_prices = np.unique(np.concatenate((buy_prices, sell_prices)))
    all_prices.sort()

    best_price = None
    best_traded = -1

    # For each candidate price p, compute:
    # demand = total buy_q with buy_price >= p
    # supply = total sell_q with sell_price <= p
    # We use vectorized searchsorted where beneficial (but loop over prices is fine here)
    # Use searchsorted to find counts quickly:
    # For buy_prices descending: count of prices >= p is k = np.sum(buy_prices >= p)
    # But we can using boolean sum (fast in numpy)
    for p in all_prices:
        # demand
        k_buy = np.searchsorted(-buy_prices, -p, side='left')  # number of buy prices >= p
        demand = cum_buy[k_buy - 1] if k_buy > 0 else 0
        # supply
        k_sell = np.searchsorted(sell_prices, p, side='right') # number of sell prices <= p
        supply = cum_sell[k_sell - 1] if k_sell > 0 else 0

        traded = min(demand, supply)
        # choose price maximizing traded volume; tie-breaker: higher price (choose p with >=)
        if traded > best_traded or (traded == best_traded and (best_price is None or p > best_price)):
            best_traded = traded
            best_price = p

    # fallback
    if best_price is None:
        return (np.mean(np.concatenate((buy_prices[:1], sell_prices[:1]))), 0)
    return float(best_price), int(best_traded)

# -------------------------
# Auction step with money pool & tax & fair value
# -------------------------
def auction_step(fair_value, money_pool, n_orders=N_ORDERS, tax_rate=TAX_RATE):
    """
    Builds orders around current fair_value, matches them, applies tax and money constraints.
    Returns:
      auction_price, cleared_volume, money_pool_after, tax_collected, buy_cumvol, sell_cumvol, buy_prices, sell_prices
    """

    # 1) build buy & sell orders (prices are noisy around fair_value)
    buys = np.random.normal(loc=fair_value, scale=0.5, size=n_orders)   # buy-side price draws
    sells = np.random.normal(loc=fair_value, scale=0.5, size=n_orders)  # sell-side price draws

    # enforce minimum positive prices
    buys = np.clip(buys, MIN_PRICE, None)
    sells = np.clip(sells, MIN_PRICE, None)

    # volumes per order drawn from sample_volume()
    buy_q = np.array([sample_volume() for _ in range(n_orders)], dtype=np.int64)
    sell_q = np.array([sample_volume() for _ in range(n_orders)], dtype=np.int64)

    # sort to order book (buys high->low, sells low->high)
    buy_idx = np.argsort(-buys)
    sell_idx = np.argsort(sells)

    buy_prices = buys[buy_idx]
    buy_q = buy_q[buy_idx]
    sell_prices = sells[sell_idx]
    sell_q = sell_q[sell_idx]

    # compute clearing price & quantity ignoring money constraint first
    auction_price, cleared_vol = find_clearing_price_and_volume(buy_prices, buy_q, sell_prices, sell_q)

    # 2) enforce aggregate money constraint: buyers cannot spend more than money_pool
    # compute total nominal buy demand at price = auction_price
    # Demand (before scaling) is total buy_q with buy_price >= auction_price
    k_buy = np.searchsorted(-buy_prices, -auction_price, side='left')
    demand = np.sum(buy_q[:k_buy]) if k_buy > 0 else 0
    # total nominal spending to execute "cleared_vol" at auction price:
    nominal_spend = auction_price * cleared_vol
    # if nominal_spend exceeds money_pool, scale down traded quantity proportionally
    if nominal_spend > 0 and nominal_spend > money_pool:
        # scale factor
        scale = money_pool / nominal_spend
        # reduce cleared volume
        scaled_traded = int(np.floor(cleared_vol * scale))
        cleared_vol = scaled_traded
        # if scaled_traded becomes 0, no trade
        if cleared_vol == 0:
            return auction_price, 0, money_pool, 0.0, np.cumsum(buy_q), np.cumsum(sell_q), buy_prices, sell_prices

    # 3) apply tax & update money pool
    trade_value = auction_price * cleared_vol
    tax = trade_value * tax_rate
    # money: buyers pay trade_value + tax; sellers receive trade_value - tax (net tax removed)
    # For aggregate pool, buyers' cash decreases by trade_value + tax.
    # Sellers receiving money increases available cash in system, but tax is removed from the system.
    # For simplicity we assume sellers' proceeds remain in the money_pool (buyers pay from pool to sellers),
    # but tax is removed from pool (treasury).
    # net effect on money_pool is -tax (since trade_value moves within system)
    money_pool_after = money_pool - tax

    # Safety: clamp to nonnegative
    money_pool_after = max(0.0, money_pool_after)

    # return arrays for plotting convenience (cumulative volumes)
    buy_cumvol = np.cumsum(buy_q)
    sell_cumvol = np.cumsum(sell_q)

    return auction_price, int(cleared_vol), money_pool_after, float(tax), buy_cumvol, sell_cumvol, buy_prices, sell_prices

# -------------------------
# Main simulation loop
# -------------------------
if __name__ == "__main__":
    steps = STEPS
    fair_value = INIT_FAIR
    money_pool = INIT_MONEY

    prices = np.zeros(steps)
    volumes = np.zeros(steps, dtype=np.int64)
    fairs = np.zeros(steps)
    money_pools = np.zeros(steps)
    taxes = np.zeros(steps)

    last_snapshot = None

    for t in tqdm(range(steps)):
        # 1) occasional news
        if np.random.rand() < NEWS_PROB:
            if np.random.rand() < NEWS_BIG_PROB:
                shock = np.random.normal(0, NEWS_SIGMA_BIG)
            else:
                shock = np.random.normal(0, NEWS_SIGMA_SMALL)
            fair_value = max(MIN_PRICE, fair_value + shock)  # news shifts fair value

        # 2) fundamental drift/noise
        fair_value = max(MIN_PRICE, fair_value + FUND_DRIFT + np.random.normal(0, FUND_SIGMA))

        # 3) run auction with current fair value
        auction_price, cleared_vol, money_pool, tax_collected, buy_cumvol, sell_cumvol, buy_prices, sell_prices = auction_step(
            fair_value, money_pool, n_orders=N_ORDERS, tax_rate=TAX_RATE
        )

        # 4) record
        prices[t] = auction_price
        volumes[t] = cleared_vol
        fairs[t] = fair_value
        money_pools[t] = money_pool
        taxes[t] = tax_collected
        last_snapshot = (auction_price, cleared_vol, buy_cumvol, sell_cumvol, buy_prices, sell_prices)

        # 5) liquidity injections occasionally
        if (t + 1) % LIQ_INJECT_FREQ == 0:
            injection = max(0.0, np.random.normal(LIQ_INJECT_MEAN, LIQ_INJECT_STD))
            money_pool += injection

    # Unpack last snapshot for final order-book plotting
    if last_snapshot is not None:
        auction_price_final, cleared_vol_final, buy_cumvol, sell_cumvol, buy_prices, sell_prices = last_snapshot
    else:
        auction_price_final = prices[-1]
        cleared_vol_final = volumes[-1]
        buy_cumvol = np.array([])
        sell_cumvol = np.array([])
        buy_prices = np.array([])
        sell_prices = np.array([])

    # -------------------------
    # Plots
    # -------------------------
    time = np.arange(steps)

    plt.figure(figsize=(10, 5))
    plt.plot(time, prices, label="Auction Price")
    plt.plot(time, fairs, label="Fair Value (fundamental)", alpha=0.7)
    plt.xlabel("Time step")
    plt.ylabel("Price")
    plt.title("Price and Fundamental over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(time, volumes, color="purple")
    plt.xlabel("Time step")
    plt.ylabel("Cleared Volume")
    plt.title("Cleared Volume (Traded) vs Time")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(time, money_pools, color="orange")
    plt.xlabel("Time step")
    plt.ylabel("Money Pool (cash)")
    plt.title("Money Pool over Time (taxes reduce pool; injections add)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(time, taxes.cumsum(), label="Cumulative Taxes Collected")
    plt.xlabel("Time step")
    plt.ylabel("Tax")
    plt.title("Cumulative Tax Collected over Time")
    plt.grid(True)
    plt.show()

    # histogram of returns
    returns = np.diff(prices)
    plt.figure(figsize=(7, 4))
    plt.hist(returns, bins=80, density=True, alpha=0.7)
    plt.title("Histogram of Price Returns (P_t - P_{t-1})")
    plt.grid(True)
    plt.show()

    # final orderbook snapshot (if available)
    if buy_cumvol.size and sell_cumvol.size:
        plt.figure(figsize=(7, 5))
        plt.step(buy_cumvol, buy_prices, where='post', color='green', label='Bids')
        plt.step(sell_cumvol, sell_prices, where='post', color='red', label='Asks')
        plt.axhline(auction_price_final, linestyle='--', color='black', label=f"Auction Price {auction_price_final:.3f}")
        plt.xlabel("Cumulative Volume")
        plt.ylabel("Price")
        plt.title("Final Order Book Snapshot")
        plt.legend()
        plt.grid(True)
        plt.show()

    print("Done. Example stats:")
    print(f"Final auction price: {prices[-1]:.4f}")
    print(f"Total traded volume: {volumes.sum():,}")
    print(f"Total tax collected: {taxes.sum():.2f}")
    print(f"Final money pool: {money_pools[-1]:.2f}")
