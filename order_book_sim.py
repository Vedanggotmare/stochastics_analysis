import numpy as np
import random
import heapq
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Market:
    def __init__(self, init_price=100):
        self.price = float(init_price)
        self.price_history = [self.price]
        self.volume_history = []
        self.trades = []         # list of list: each element = [(trade_price, trade_volume), ...]
        self.order_books = []    # list of tuples: (buy_depth_dict, sell_depth_dict)
        self.news_events = {}    # mapping: step -> {"type": "positive"/"negative"/"neutral", "strength": float}

    def add_news_event(self, step, news_type="positive", strength=1.0, label=None):
        """
        Register a news event at time `step`.
        news_type: "positive", "negative", or "neutral"
        strength: how strong the effect is (float >= 0)
        label: optional string used when plotting
        """
        self.news_events[step] = {"type": news_type, "strength": float(strength), "label": label or news_type}

    def generate_orders(self, n_orders=50, dist="normal", step=0):
        orders = []
        news = self.news_events.get(step, None)

        for _ in range(n_orders):
            # Base order type (50/50)
            order_type = random.choice(["buy", "sell"])

            # Apply news bias to order direction probabilistically
            if news:
                typ = news["type"]
                if typ == "positive":
                    order_type = random.choices(["buy", "sell"], weights=[0.7, 0.3])[0]
                elif typ == "negative":
                    order_type = random.choices(["buy", "sell"], weights=[0.3, 0.7])[0]
                else:  # neutral -> no direction bias but higher activity / volatility
                    order_type = random.choices(["buy", "sell"], weights=[0.5, 0.5])[0]

            # Volume distribution
            if dist == "normal":
                volume = max(1, int(np.random.normal(50, 10)))
            elif dist == "exponential":
                volume = max(1, int(np.random.exponential(30)))
            else:
                volume = random.randint(1, 100)

            # Increase volume during news depending on strength
            if news:
                volume = max(1, int(volume * (1 + 0.6 * news["strength"])))

            # Price deviation
            deviation = np.random.normal(0, 2)
            if news:
                s = news["strength"]
                if news["type"] == "positive" and order_type == "buy":
                    # shift buying orders higher
                    deviation += abs(np.random.normal(2.0 * s, 1.0))
                elif news["type"] == "negative" and order_type == "sell":
                    # shift selling orders lower
                    deviation -= abs(np.random.normal(2.0 * s, 1.0))
                elif news["type"] == "neutral":
                    # pure volatility spike
                    deviation = np.random.normal(0, 5.0 * s)

            price = max(0.01, self.price + deviation)
            orders.append((order_type, volume, price))

        return orders

    def match_orders(self, orders):
        # Convert orders into buy (max-heap) and sell (min-heap)
        buys = [(-o[2], o[1]) for o in orders if o[0] == "buy"]
        sells = [(o[2], o[1]) for o in orders if o[0] == "sell"]
        heapq.heapify(buys)
        heapq.heapify(sells)

        trades = []
        total_volume = 0

        # Match top-of-book while prices cross
        while buys and sells:
            buy_price, buy_vol = buys[0]
            sell_price, sell_vol = sells[0]
            buy_price_pos = -buy_price

            if buy_price_pos >= sell_price:
                traded_volume = min(buy_vol, sell_vol)
                trade_price = (buy_price_pos + sell_price) / 2.0
                trades.append((trade_price, traded_volume))
                total_volume += traded_volume

                # update buy
                if buy_vol > traded_volume:
                    heapq.heapreplace(buys, (-buy_price_pos, buy_vol - traded_volume))
                else:
                    heapq.heappop(buys)

                # update sell
                if sell_vol > traded_volume:
                    heapq.heapreplace(sells, (sell_price, sell_vol - traded_volume))
                else:
                    heapq.heappop(sells)
            else:
                break

        # Update market price to last trade price if trades occurred
        if trades:
            self.price = trades[-1][0]
        # Append histories
        self.price_history.append(self.price)
        self.volume_history.append(total_volume)
        self.trades.append(trades)

        # Build simplified depth snapshots from remaining heaps
        buy_depth = {}
        # buys heap stores (-price, vol)
        for negp, v in buys:
            p = -negp
            buy_depth[p] = buy_depth.get(p, 0) + v
        sell_depth = {}
        for p, v in sells:
            sell_depth[p] = sell_depth.get(p, 0) + v
        self.order_books.append((buy_depth, sell_depth))

    def run(self, steps=100, orders_per_step=50, dist="normal"):
        for step in tqdm(range(steps), desc="Simulating Market"):
            orders = self.generate_orders(n_orders=orders_per_step, dist=dist, step=step)
            self.match_orders(orders)
        return self.price_history, self.volume_history, self.trades, self.order_books


def animate_market(market: Market, interval=100):
    prices = market.price_history
    volumes = market.volume_history
    trades = market.trades
    order_books = market.order_books
    news_events = market.news_events  # dict: step -> info

    n_frames = len(prices)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[3, 0.9, 0.9])
    ax_price = fig.add_subplot(gs[0, 0])
    ax_volume = fig.add_subplot(gs[1, 0], sharex=ax_price)
    ax_book = fig.add_subplot(gs[:, 1])
    ax_legend = fig.add_subplot(gs[:, 2])
    ax_legend.axis("off")

    # Price line
    price_line, = ax_price.plot([], [], lw=1.5)
    ax_price.set_title("Stock Price Evolution")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle="--", alpha=0.4)

    # scatter for trades (x = time step index, y = trade price)
    trade_scat = ax_price.scatter([], [], s=[], alpha=0.7)

    # Volume bars (we'll update by clearing bars in update)
    ax_volume.set_title("Trade Volumes")
    ax_volume.set_xlabel("Time Step")
    ax_volume.set_ylabel("Volume")

    # Prepare news vertical lines once
    news_lines = {}
    for step, info in news_events.items():
        vline = ax_price.axvline(step, linestyle="--", linewidth=1.2,
                                 label=f"{info.get('label', info['type']).capitalize()} (t={step})")
        news_lines[step] = vline
        # initial set invisible; visibility toggled in update
        vline.set_visible(False)

    # Legend panel - show small legend keys (price, up-trade, down-trade, news)
    ax_legend.text(0.05, 0.9, "Legend:", fontsize=10, weight="bold")
    ax_legend.scatter([], [], color="green", label="Up trade (priceâ)", s=40)
    ax_legend.scatter([], [], color="red", label="Down trade (priceâ)", s=40)
    ax_legend.plot([], [], color="cyan", label="Price")
    ax_legend.axis("off")

    # Order book settings
    ax_book.set_title("Supply vs Demand (Order Book Snapshot)")
    ax_book.set_xlabel("Cumulative Volume")
    ax_book.set_ylabel("Price")

    # For dynamic scatter update
    trade_xs = []
    trade_ys = []
    trade_sizes = []
    trade_colors = []

    def update(frame):
        # frame is the number of points to show (1..n_frames-1)
        if frame < 1:
            return price_line, trade_scat

        # Price line update
        x = list(range(frame))
        y = prices[:frame]
        price_line.set_data(x, y)
        ax_price.set_xlim(0, max(10, n_frames))
        # dynamic y-limits with padding
        ymin = min(prices[:max(1, frame)]) * 0.98
        ymax = max(prices[:max(1, frame)]) * 1.02
        if ymin == ymax:
            ymin -= 1
            ymax += 1
        ax_price.set_ylim(ymin, ymax)

        # Update trades markers: append trades that happened at time (frame-1)
        t_index = frame - 1
        if t_index < len(trades):
            tlist = trades[t_index]
            for (tp, tv) in tlist:
                trade_xs.append(t_index)
                trade_ys.append(tp)
                # size scaling (clamped)
                size = max(10, min(400, int(2 * tv)))
                trade_sizes.append(size)
                # choose color based on whether trade moved price up vs previous price
                prev_price = prices[max(0, t_index - 1)]
                color = "green" if tp > prev_price else "red"
                trade_colors.append(color)

        if trade_xs:
            offsets = np.column_stack((trade_xs, trade_ys))
            trade_scat.set_offsets(offsets)
            trade_scat.set_sizes(trade_sizes)
            trade_scat.set_color(trade_colors)
        else:
            trade_scat.set_offsets([])

        # Volume bar update (clear+redraw is simplest)
        ax_volume.clear()
        ax_volume.bar(range(frame - 1), volumes[:frame - 1], alpha=0.8)
        ax_volume.set_xlim(0, max(10, n_frames))
        ax_volume.set_ylabel("Volume")
        ax_volume.set_xlabel("Time Step")
        ax_volume.grid(True, linestyle="--", alpha=0.3)

        # Order book snapshot for this frame (if available)
        ax_book.clear()
        if t_index < len(order_books):
            buy_depth, sell_depth = order_books[t_index]
            # convert to cumulative depth sorted by price
            if buy_depth:
                # sort buy prices descending for cumulative demand curve
                bp_sorted = sorted(buy_depth.items(), key=lambda x: -x[0])
                buy_prices = [p for p, v in bp_sorted]
                buy_vols = np.cumsum([v for p, v in bp_sorted])
                ax_book.step(buy_vols, buy_prices, where="post", label="Demand")
            if sell_depth:
                sp_sorted = sorted(sell_depth.items(), key=lambda x: x[0])
                sell_prices = [p for p, v in sp_sorted]
                sell_vols = np.cumsum([v for p, v in sp_sorted])
                ax_book.step(sell_vols, sell_prices, where="post", label="Supply")
            ax_book.set_xlabel("Cumulative Volume")
            ax_book.set_ylabel("Price")
            ax_book.legend()
            ax_book.grid(True, linestyle="--", alpha=0.3)

        # Toggle visibility of news vertical lines at the appropriate time (keep visible from their step onward)
        for step, vline in news_lines.items():
            if frame > step:
                vline.set_visible(True)
                # Add small label near top of plot for the news event (once)
                info = news_events[step]
                label = info.get("label", info["type"]).capitalize()
                # place label if not already present; we'll attempt to draw text at the top
                # To avoid duplicate texts, check existing texts:
                existing_labels = [t.get_text() for t in ax_price.texts]
                txt = f"{label} (t={step})"
                if txt not in existing_labels:
                    ax_price.text(step + 0.5, ax_price.get_ylim()[1] * 0.98, txt, rotation=90,
                                  va="top", fontsize=8, alpha=0.8)

        return price_line, trade_scat

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    market = Market(init_price=100.0)

    # Schedule some news events: (step index, type, strength, label)
    market.add_news_event(step=30, news_type="positive", strength=1.2, label="Earnings beat")
    market.add_news_event(step=75, news_type="neutral", strength=1.0, label="Fed minutes")
    market.add_news_event(step=130, news_type="negative", strength=1.8, label="Product recall")
    market.add_news_event(step=160, news_type="positive", strength=0.8, label="Acquisition rumor")

    # Run simulation
    steps = 200
    prices, volumes, trades, order_books = market.run(steps=steps, orders_per_step=120, dist="normal")

    # Animate results (this will open a matplotlib window)
    animate_market(market, interval=100)