# market_sim_agents_numba.py
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# try to import numba
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(fn=None, **kwargs):
        return fn

# -------------------------
# CONFIG
# -------------------------
STEPS = 5000
N_ORDERS_PER_STEP = 300         # orders drawn per side potential (not final matched)
N_AGENTS = 200                  # number of agents
ORDER_BOOK_CAP = 50000          # max live orders maintained
INIT_FAIR = 10.0
INIT_MONEY_POOL = 1_000_000.0
TAX_RATE = 0.001
LIQ_INJECT_FREQ = 500
LIQ_INJECT_MEAN = 50_000
NEWS_PROB = 0.02
NEWS_SIGMA_SMALL = 0.2
NEWS_BIG_PROB = 0.01
NEWS_SIGMA_BIG = 2.0
FUND_DRIFT = 0.0
FUND_SIGMA = 0.05
MIN_PRICE = 0.01

# agent behavior params
P_MARKET = 0.15         # probability an agent uses market order instead of limit
P_SUBMIT = 0.25         # probability an agent submits an order at a given step
MEAN_ORDER_Q = 20
MAX_ORDER_Q = 100000
CANCEL_PROB = 0.02      # probability per existing order to be canceled each step
ORDER_LIFETIME_MEAN = 50  # geometric mean lifetime of an order (steps)

SEED = 49
np.random.seed(SEED)
random.seed(SEED)

# -------------------------
# helper functions
# -------------------------
def sample_volume_lognormal():
    if np.random.rand() < 0.97:
        v = int(np.random.lognormal(mean=3.0, sigma=0.6))
    else:
        v = int(1 + np.random.pareto(2.2) * 100)
    return max(1, min(v, MAX_ORDER_Q))

def clip_price(x):
    return max(MIN_PRICE, float(x))

# -------------------------
# Orderbook arrays
# -------------------------
# We'll maintain arrays of length ORDER_BOOK_CAP and a pointer 'n_orders'
# Fields: price, qty, side (1=buy, -1=sell), owner (agent id), expiry (step), active (0/1)
order_price = np.zeros(ORDER_BOOK_CAP, dtype=np.float64)
order_qty = np.zeros(ORDER_BOOK_CAP, dtype=np.int64)
order_side = np.zeros(ORDER_BOOK_CAP, dtype=np.int8)
order_owner = np.full(ORDER_BOOK_CAP, -1, dtype=np.int32)
order_expiry = np.zeros(ORDER_BOOK_CAP, dtype=np.int32)
order_active = np.zeros(ORDER_BOOK_CAP, dtype=np.int8)
n_orders = 0
free_idx_stack = []  # free indices for reuse (python list ok)

# -------------------------
# Agent arrays
# -------------------------
agent_cash = np.full(N_AGENTS, INIT_MONEY_POOL / N_AGENTS, dtype=np.float64)
agent_inventory = np.zeros(N_AGENTS, dtype=np.int64)

# -------------------------
# Numba-accelerated matching
# -------------------------
# We'll implement a function that:
#  - takes arrays of active orders (price, qty, side)
#  - computes clearing price (from unique prices)
#  - determines cleared quantity
#  - allocates filled quantities to each order pro-rata among orders eligible at clearing price
# Returns: auction_price (float), total_traded (int), fills array (same length) indicating qty executed per order
#
# Implementation notes: numba doesn't support advanced numpy like np.unique easily; we implement with loops.

@njit
def find_clearing_and_allocate_nb(prices, qtys, sides, active_mask):
    # prices, qtys, sides, active_mask are 1D arrays of same length
    L = prices.shape[0]
    # gather active indices
    # first count actives
    count = 0
    for i in range(L):
        if active_mask[i]:
            count += 1
    if count == 0:
        return 0.0, 0, np.zeros(L, dtype=np.int64)

    # create compact arrays
    p = np.empty(count, dtype=np.float64)
    q = np.empty(count, dtype=np.int64)
    s = np.empty(count, dtype=np.int8)
    idxmap = np.empty(count, dtype=np.int64)
    k = 0
    for i in range(L):
        if active_mask[i]:
            p[k] = prices[i]
            q[k] = qtys[i]
            s[k] = sides[i]
            idxmap[k] = i
            k += 1

    # separate buys and sells
    # count buys/sells
    nb = 0
    ns = 0
    for i in range(count):
        if s[i] == 1:
            nb += 1
        else:
            ns += 1

    buy_p = np.empty(nb, dtype=np.float64)
    buy_q = np.empty(nb, dtype=np.int64)
    buy_idx = np.empty(nb, dtype=np.int64)
    sell_p = np.empty(ns, dtype=np.float64)
    sell_q = np.empty(ns, dtype=np.int64)
    sell_idx = np.empty(ns, dtype=np.int64)
    ib = 0
    is_ = 0
    for i in range(count):
        if s[i] == 1:
            buy_p[ib] = p[i]
            buy_q[ib] = q[i]
            buy_idx[ib] = idxmap[i]
            ib += 1
        else:
            sell_p[is_] = p[i]
            sell_q[is_] = q[i]
            sell_idx[is_] = idxmap[i]
            is_ += 1

    # sort buys desc, sells asc (simple bubble/selection sorts will be slow; use argsort via numpy not supported directly in njit)
    # use simple insertion sort for moderate sizes
    # sort buys by price descending
    for i in range(nb):
        for j in range(i+1, nb):
            if buy_p[j] > buy_p[i]:
                tmp = buy_p[i]; buy_p[i] = buy_p[j]; buy_p[j] = tmp
                tmpq = buy_q[i]; buy_q[i] = buy_q[j]; buy_q[j] = tmpq
                tmpi = buy_idx[i]; buy_idx[i] = buy_idx[j]; buy_idx[j] = tmpi
    # sort sells by price ascending
    for i in range(ns):
        for j in range(i+1, ns):
            if sell_p[j] < sell_p[i]:
                tmp = sell_p[i]; sell_p[i] = sell_p[j]; sell_p[j] = tmp
                tmpq = sell_q[i]; sell_q[i] = sell_q[j]; sell_q[j] = tmpq
                tmpi = sell_idx[i]; sell_idx[i] = sell_idx[j]; sell_idx[j] = tmpi

    # build candidate prices (unique union)
    # naive merging since sorted
    i = 0; j = 0; m = 0
    # upper bound size
    cap = nb + ns
    cand = np.empty(cap, dtype=np.float64)
    last = -1e100
    while i < nb or j < ns:
        val = 0.0
        if i < nb and (j >= ns or buy_p[i] < sell_p[j]):
            val = buy_p[i]
            i += 1
        else:
            val = sell_p[j]
            j += 1
        # ensure uniqueness
        if m == 0 or val != last:
            cand[m] = val
            last = val
            m += 1
    # if none put a middle price
    if m == 0:
        # fallback
        mid = 1.0 if (nb+ns)==0 else 0.5*(buy_p[0] if nb>0 else 1.0 + sell_p[0] if ns>0 else 1.0)
        return mid, 0, np.zeros(L, dtype=np.int64)

    best_price = 0.0
    best_traded = -1
    # precompute buy cumulative and sell cumulative for quick access
    # But as candidate prices are sorted ascending (due to merging), we compute demand and supply using loops
    for ii in range(m):
        pval = cand[ii]
        # demand: sum of buy_q where buy_p >= pval
        demand = 0
        for t in range(nb):
            if buy_p[t] >= pval:
                demand += buy_q[t]
            else:
                break
        # supply: sum of sell_q where sell_p <= pval
        supply = 0
        for t in range(ns):
            if sell_p[t] <= pval:
                supply += sell_q[t]
            else:
                break
        traded = demand if demand < supply else supply
        # choose price maximizing traded; tie-breaker prefer higher price
        if traded > best_traded or (traded == best_traded and pval > best_price):
            best_traded = traded
            best_price = pval

    # allocate fills pro-rata among eligible orders at best_price
    fills = np.zeros(L, dtype=np.int64)
    if best_traded <= 0:
        return best_price, 0, fills

    # eligible buy orders indices and total demand_eligible
    demand_eligible = 0
    for t in range(nb):
        if buy_p[t] >= best_price:
            demand_eligible += buy_q[t]
        else:
            break
    supply_eligible = 0
    for t in range(ns):
        if sell_p[t] <= best_price:
            supply_eligible += sell_q[t]
        else:
            break

    # traded quantity best_traded splits: we will fill proportionally on the smaller-side? In auction, both sides matched to same quantity.
    # We'll allocate to both sides pro-rata to their eligible quantity.
    # compute buy_fill_factor and sell_fill_factor
    if demand_eligible == 0 or supply_eligible == 0:
        return best_price, 0, fills

    # allocate: buyer side allocated = min(demand_eligible, best_traded) proportionally
    # but ensure total allocated equals best_traded on each side; we distribute best_traded proportionally
    # For buyers: fill_i = floor(buy_q_i * best_traded / demand_eligible)
    acc = 0
    for t in range(nb):
        if buy_p[t] >= best_price:
            take = (buy_q[t] * best_traded) // demand_eligible
            fills[buy_idx[t]] = take
            acc += take
        else:
            break
    # fix remainder by adding 1 to earliest eligible until sum == best_traded
    rem = best_traded - acc
    tt = 0
    while rem > 0:
        if tt >= nb:
            break
        if buy_p[tt] >= best_price:
            fills[buy_idx[tt]] += 1
            rem -= 1
        tt += 1

    # sellers
    acc = 0
    for t in range(ns):
        if sell_p[t] <= best_price:
            take = (sell_q[t] * best_traded) // supply_eligible
            fills[sell_idx[t]] = take
            acc += take
        else:
            break
    rem = best_traded - acc
    tt = 0
    while rem > 0:
        if tt >= ns:
            break
        if sell_p[tt] <= best_price:
            fills[sell_idx[tt]] += 1
            rem -= 1
        tt += 1

    # ensure fills array sums to something sensible (ideally both buy and sell sums == best_traded)
    return best_price, best_traded, fills

# If numba not available, provide numpy fallback (similar logic but likely slower)
def find_clearing_and_allocate_np(prices, qtys, sides, active_mask):
    # build lists
    idx = np.where(active_mask == 1)[0]
    if idx.size == 0:
        return 0.0, 0, np.zeros_like(prices, dtype=np.int64)
    p = prices[idx]; q = qtys[idx]; s = sides[idx]; imap = idx
    buy_mask = (s == 1); sell_mask = (s == -1)
    buy_p = p[buy_mask]; buy_q = q[buy_mask]; buy_i = imap[buy_mask]
    sell_p = p[sell_mask]; sell_q = q[sell_mask]; sell_i = imap[sell_mask]
    # sort
    buy_order = np.argsort(-buy_p); sell_order = np.argsort(sell_p)
    buy_p = buy_p[buy_order]; buy_q = buy_q[buy_order]; buy_i = buy_i[buy_order]
    sell_p = sell_p[sell_order]; sell_q = sell_q[sell_order]; sell_i = sell_i[sell_order]
    all_prices = np.unique(np.concatenate((buy_p, sell_p)))
    best_price = None; best_traded = -1
    for pval in all_prices:
        demand = buy_q[buy_p >= pval].sum()
        supply = sell_q[sell_p <= pval].sum()
        traded = min(demand, supply)
        if traded > best_traded or (traded == best_traded and (best_price is None or pval > best_price)):
            best_traded = traded
            best_price = pval
    if best_traded <= 0:
        return best_price if best_price is not None else 0.0, 0, np.zeros_like(prices, dtype=np.int64)
    # allocate proportional fills
    fills = np.zeros_like(prices, dtype=np.int64)
    demand_eligible = buy_q[buy_p >= best_price].sum()
    supply_eligible = sell_q[sell_p <= best_price].sum()
    # buyers
    if demand_eligible > 0:
        allocated = np.floor(buy_q * best_traded / demand_eligible).astype(np.int64)
        # only for eligible indices
        mask = buy_p >= best_price
        fills[buy_i[mask]] = allocated[mask]
        rem = best_traded - fills.sum()
        # fix remainder
        j = 0
        while rem > 0 and j < buy_i[mask].size:
            fills[buy_i[mask][j]] += 1
            rem -= 1
            j += 1
    # sellers
    # For simplicity ensure symmetry by capping sellers to buyers in fills; but keep naive allocation
    return best_price, best_traded, fills

# choose implementation
if NUMBA_OK:
    find_clearing_and_allocate = find_clearing_and_allocate_nb
else:
    find_clearing_and_allocate = find_clearing_and_allocate_np

# -------------------------
# Utility: add order into book
# -------------------------
def add_order(price, qty, side, owner, expiry, step):
    global n_orders
    # reuse free index if available
    if free_idx_stack:
        idx = free_idx_stack.pop()
    else:
        if n_orders >= ORDER_BOOK_CAP:
            # if full, do simple replacement: overwrite some random inactive slot - but to keep simple skip adding
            # find first inactive slot
            found = False
            for j in range(ORDER_BOOK_CAP):
                if order_active[j] == 0:
                    idx = j
                    found = True
                    break
            if not found:
                return -1
        else:
            idx = n_orders
            n_orders += 1
    order_price[idx] = float(price)
    order_qty[idx] = int(qty)
    order_side[idx] = int(side)
    order_owner[idx] = int(owner)
    order_expiry[idx] = int(step + expiry)
    order_active[idx] = 1
    return idx

# remove order
def remove_order(idx):
    order_active[idx] = 0
    order_owner[idx] = -1
    # push to free stack
    free_idx_stack.append(int(idx))

# -------------------------
# Agent decision function
# -------------------------
def agent_decide_and_submit(agent_id, fair_value, last_price, step):
    # returns list of orders to submit: (price, qty, side, owner, expiry)
    res = []
    if np.random.rand() > P_SUBMIT:
        return res
    # decide buy or sell: slightly bias toward fundamental (if fair_value > last_price => more buy)
    diff = fair_value - last_price
    prob_buy = 0.5 + 0.1 * np.tanh(diff / max(1e-6, last_price))
    if np.random.rand() < prob_buy:
        side = 1
    else:
        side = -1
    qty = sample_volume_lognormal()
    if np.random.rand() < P_MARKET:
        # market order: represent with extreme price so it's included at clearing
        price = fair_value + (0.5 * last_price * (1.0 if side==1 else -1.0)) + (np.random.randn() * 0.01)
        # extreme direction to ensure execution
        price = clip_price(price + (5.0 if side==1 else -5.0))
        expiry = 1   # immediate - will either execute or die next step
    else:
        # limit price around fair_value with some spread
        price = fair_value + np.random.normal(0, 0.3)
        price = clip_price(price)
        # lifetime geometric ~ mean ORDER_LIFETIME_MEAN
        expiry = max(1, int(np.random.geometric(1.0/ORDER_LIFETIME_MEAN)))
    res.append((price, qty, side, agent_id, expiry))
    return res

# -------------------------
# Main simulation
# -------------------------
if __name__ == "__main__":
    steps = STEPS
    fair_value = INIT_FAIR
    money_pool = INIT_MONEY_POOL
    last_price = fair_value

    prices = np.zeros(steps)
    volumes = np.zeros(steps, dtype=np.int64)
    fairs = np.zeros(steps)
    money_pools = np.zeros(steps)
    taxes = np.zeros(steps)

    # bookkeeping: optionally compute agent wealth distribution periodically
    for t in tqdm(range(steps)):
        # news
        if np.random.rand() < NEWS_PROB:
            if np.random.rand() < NEWS_BIG_PROB:
                shock = np.random.normal(0, NEWS_SIGMA_BIG)
            else:
                shock = np.random.normal(0, NEWS_SIGMA_SMALL)
            fair_value = max(MIN_PRICE, fair_value + shock)
        # fundamental drift
        fair_value = max(MIN_PRICE, fair_value + FUND_DRIFT + np.random.normal(0, FUND_SIGMA))

        # cancellations: random cancellation of active orders
        # iterate a random subset for performance
        if n_orders > 0:
            # pick some indices to test
            idxs = np.random.randint(0, max(1, n_orders), size=min(200, max(1, n_orders)))
            for idx in np.unique(idxs):
                if order_active[idx] == 1:
                    # expire by time
                    if order_expiry[idx] <= t:
                        remove_order(idx)
                    else:
                        if np.random.rand() < CANCEL_PROB:
                            remove_order(idx)

        # agents submit orders
        for agent_id in range(N_AGENTS):
            orders = agent_decide_and_submit(agent_id, fair_value, last_price, t)
            for (price, qty, side, owner, expiry) in orders:
                # check agent cash constraint for buy side (quick check)
                if side == 1:
                    # rough check: cannot spend more than cash
                    max_affordable = int(agent_cash[owner] // price) if price>0 else 0
                    if max_affordable <= 0:
                        continue
                    # cap qty
                    qty = min(qty, max_affordable)
                add_order(price, qty, side, owner, expiry, step=t)

        # prepare arrays for matching (call numba function)
        prices_arr = order_price.copy()
        qtys_arr = order_qty.copy()
        sides_arr = order_side.copy()
        active_mask = order_active.copy()

        # find clearing price and allocate fills
        auction_price, total_traded, fills = find_clearing_and_allocate(prices_arr, qtys_arr, sides_arr, active_mask)

        # enforce money_pool constraint: buyers cannot spend more than money_pool
        trade_value = auction_price * total_traded
        if trade_value > money_pool and trade_value > 0:
            scale = money_pool / trade_value
            scaled_traded = int(np.floor(total_traded * scale))
            total_traded = scaled_traded
            # scale fills proportionally
            if total_traded == 0:
                fills *= 0
            else:
                # proportionally reduce
                current_sum = fills.sum()
                if current_sum > 0:
                    factor = total_traded / current_sum
                    new_fills = (fills.astype(np.float64) * factor).astype(np.int64)
                    # fix remainder
                    rem = total_traded - new_fills.sum()
                    i = 0
                    L = new_fills.shape[0]
                    while rem > 0 and i < L:
                        if new_fills[i] < qtys_arr[i] and active_mask[i]:
                            new_fills[i] += 1
                            rem -= 1
                        i += 1
                    fills = new_fills

        # compute tax and update money pool & transfer cash and inventory per agent
        tax = auction_price * fills.sum() * TAX_RATE
        # buyers pay auction_price * filled + tax; sellers receive auction_price * filled - tax
        # For simplicity, we assume buyer cash decreases, seller cash increases; tax removed from system.
        # Fill processing
        # iterate orders and update owners
        for i in range(n_orders):
            if i >= fills.shape[0]:
                break
            f = fills[i]
            if f <= 0:
                continue
            owner = int(order_owner[i])
            side = int(order_side[i])
            if side == 1:
                # buyer: reduce cash, increase inventory
                cost = auction_price * f
                fee = cost * TAX_RATE
                total_cost = cost + fee
                # guard if agent lacks cash, skip fill (shouldn't happen due to earlier constraints)
                if agent_cash[owner] + 1e-9 >= total_cost:
                    agent_cash[owner] -= total_cost
                    agent_inventory[owner] += f
                else:
                    # if underfunded, skip (decrease f to zero)
                    # safe fallback: set no transfer (we already scaled by money_pool)
                    pass
            else:
                # seller: get proceeds minus tax, reduce inventory
                proceeds = auction_price * f
                fee = proceeds * TAX_RATE
                agent_cash[owner] += (proceeds - fee)
                agent_inventory[owner] -= f

            # reduce or remove order qty
            order_qty[i] -= f
            if order_qty[i] <= 0:
                remove_order(i)

        # update money pool: tax removed
        money_pool = max(0.0, money_pool - tax)

        # occasional liquidity injection
        if (t + 1) % LIQ_INJECT_FREQ == 0:
            injection = max(0.0, np.random.normal(LIQ_INJECT_MEAN, LIQ_INJECT_MEAN * 0.1))
            money_pool += injection

        # record
        prices[t] = auction_price
        volumes[t] = fills.sum()
        fairs[t] = fair_value
        money_pools[t] = money_pool
        taxes[t] = tax
        last_price = auction_price

    # -------------------------
    # Post-simulation plots
    # -------------------------
    time = np.arange(steps)
    plt.figure(figsize=(10,5))
    plt.plot(time, prices, label="Auction Price")
    plt.plot(time, fairs, label="Fair Value", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(); plt.grid(True); plt.title("Price & Fair Value")
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(time, volumes); plt.title("Cleared Volume (sum of fills)"); plt.grid(True); plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(time, money_pools); plt.title("Money Pool"); plt.grid(True); plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(time, np.cumsum(taxes)); plt.title("Cumulative Tax Collected"); plt.grid(True); plt.show()

    # agent wealth histogram
    agent_wealth = agent_cash + agent_inventory * prices[-1]
    plt.figure(figsize=(8,4))
    plt.hist(agent_wealth, bins=40); plt.title("Agent Wealth Distribution"); plt.grid(True); plt.show()

    # final order book snapshot (aggregate)
    active_idx = np.where(order_active[:n_orders] == 1)[0]
    if active_idx.size > 0:
        buy_mask = order_side[active_idx] == 1
        sell_mask = order_side[active_idx] == -1
        buy_prices = order_price[active_idx][buy_mask]
        buy_q = order_qty[active_idx][buy_mask]
        sell_prices = order_price[active_idx][sell_mask]
        sell_q = order_qty[active_idx][sell_mask]
        if buy_prices.size > 0 and sell_prices.size > 0:
            buy_order = np.argsort(-buy_prices)
            sell_order = np.argsort(sell_prices)
            bp = buy_prices[buy_order]; bq = np.cumsum(buy_q[buy_order])
            sp = sell_prices[sell_order]; sq = np.cumsum(sell_q[sell_order])
            plt.figure(figsize=(7,5))
            plt.step(bq, bp, where='post', color='green', label='Bids')
            plt.step(sq, sp, where='post', color='red', label='Asks')
            plt.axhline(prices[-1], linestyle='--', color='black', label=f"Auction Price {prices[-1]:.3f}")
            plt.xlabel("Cumulative Volume")
            plt.ylabel("Price")
            plt.legend(); plt.grid(True); plt.title("Final Order Book Snapshot")
            plt.show()

    print("Done.")
    print(f"Final price: {prices[-1]:.4f}")
    print(f"Total traded volume (sum over steps): {volumes.sum()}")
    print(f"Total tax collected: {taxes.sum():.2f}")
    print(f"Final money pool: {money_pools[-1]:.2f}")
    print(f"Agents with negative cash/inventory (checks): min cash {agent_cash.min():.2f}, min inventory {agent_inventory.min()}")

