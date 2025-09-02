# 📈 Market Microstructure Simulator

This project simulates a simple **order-driven financial market** with news-driven dynamics, order books, trades, and animations.
It models how buy/sell orders interact, how trades occur, and how **market price evolves over time under news shocks**.

The simulation generates:

* Stock price history 📊
* Trade volumes 📦
* Trades with prices & volumes 💹
* Order book snapshots (supply vs demand curves) 📉📈
* News events affecting bias, volatility, and volume 📰

A built-in animation visualizes the evolution step by step.

---

## 🚀 Features

* **Order Generation**

  * Random buy/sell orders with configurable distributions (normal, exponential, uniform).
  * News events bias order direction and volume.

* **Order Matching**

  * Continuous double-auction style matching with buy/sell books.
  * Trades executed when best buy ≥ best sell.
  * Market price updated to the last trade price.

* **Market Dynamics**

  * Tracks price history, trade volume, executed trades, and order books.
  * Allows injecting **positive, negative, or neutral news events** with adjustable strength.

* **Visualization (Matplotlib Animation)**

  * **Top panel:** Price evolution + trade markers.
  * **Bottom panel:** Trade volumes per step.
  * **Right panel:** Order book (supply & demand curves).
  * **News events**: vertical lines + labels marking shocks.
  * Color-coded trades (green = price uptick, red = price downtick).

---

## 📂 Project Structure

```
order_book_sim.py    # Main simulation and visualization code
```

---

## ⚙️ Installation

1. Clone this repo:

   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. Install dependencies:

   ```bash
   pip install numpy matplotlib tqdm
   ```

---

## ▶️ Usage

Run the simulation:

```bash
python market_simulation.py
```

### Example in `__main__`:

```python
market = Market(init_price=100.0)

# Schedule some news events
market.add_news_event(step=30, news_type="positive", strength=1.2, label="Earnings beat")
market.add_news_event(step=75, news_type="neutral", strength=1.0, label="Fed minutes")
market.add_news_event(step=130, news_type="negative", strength=1.8, label="Product recall")
market.add_news_event(step=160, news_type="positive", strength=0.8, label="Acquisition rumor")

# Run market simulation
prices, volumes, trades, order_books = market.run(steps=200, orders_per_step=120)

# Animate results
animate_market(market, interval=100)
```

This opens a **Matplotlib window** showing price, volumes, trades, order book, and news shocks.

---

## 📊 Example Output

* Price evolution curve with trade markers (green/red).
* Volume bars per step.
* Order book demand vs supply curves.
* News events shown as vertical dashed lines with labels.

*(Animation pops up when running the script — not a static plot.)*

---

## 🧩 Customization

* **Initial Price**: `Market(init_price=...)`
* **Steps**: `market.run(steps=200, ...)`
* **Orders per Step**: `orders_per_step=120`
* **Order Distribution**: `"normal"`, `"exponential"`, `"uniform"`
* **News Events**:

  ```python
  market.add_news_event(step=50, news_type="positive", strength=1.5, label="Big earnings")
  ```

---

## 📌 Future Extensions

* Add **multiple stocks / correlated assets**.
* Implement **market makers** vs. noise traders.
* Simulate **bid-ask spread & slippage**.
* Export results to CSV for backtesting.

---
