import heapq
import random

class Market:
    def __init__(self):
        self.bids = []  # max-heap (store as -price)
        self.asks = []  # min-heap
        self.last_trade = None
        self.trade_log = []

    def insert_limit_order(self, price, volume, side):
        """Insert a limit order and attempt matching"""
        if side == 'buy':
            while self.asks and price >= self.asks[0][0] and volume > 0:
                best_ask, ask_vol = heapq.heappop(self.asks)
                trade_vol = min(volume, ask_vol)
                self.last_trade = best_ask
                self.trade_log.append((best_ask, trade_vol, 'buy'))
                volume -= trade_vol
                ask_vol -= trade_vol
                if ask_vol > 0:
                    heapq.heappush(self.asks, (best_ask, ask_vol))
            if volume > 0:
                heapq.heappush(self.bids, (-price, volume))

        elif side == 'sell':
            while self.bids and price <= -self.bids[0][0] and volume > 0:
                best_bid, bid_vol = heapq.heappop(self.bids)
                best_bid = -best_bid
                trade_vol = min(volume, bid_vol)
                self.last_trade = best_bid
                self.trade_log.append((best_bid, trade_vol, 'sell'))
                volume -= trade_vol
                bid_vol -= trade_vol
                if bid_vol > 0:
                    heapq.heappush(self.bids, (-best_bid, bid_vol))
            if volume > 0:
                heapq.heappush(self.asks, (price, volume))

    def insert_market_order(self, volume, side):
        """Market order executes fully against best available prices"""
        if side == 'buy':
            while self.asks and volume > 0:
                best_ask, ask_vol = heapq.heappop(self.asks)
                trade_vol = min(volume, ask_vol)
                self.last_trade = best_ask
                self.trade_log.append((best_ask, trade_vol, 'buy'))
                volume -= trade_vol
                ask_vol -= trade_vol
                if ask_vol > 0:
                    heapq.heappush(self.asks, (best_ask, ask_vol))
        elif side == 'sell':
            while self.bids and volume > 0:
                best_bid, bid_vol = heapq.heappop(self.bids)
                best_bid = -best_bid
                trade_vol = min(volume, bid_vol)
                self.last_trade = best_bid
                self.trade_log.append((best_bid, trade_vol, 'sell'))
                volume -= trade_vol
                bid_vol -= trade_vol
                if bid_vol > 0:
                    heapq.heappush(self.bids, (-best_bid, bid_vol))

    def cancel_order(self, price, side):
        """Very simple cancellation: just removes one order at given price"""
        if side == 'buy':
            self.bids = [o for o in self.bids if -o[0] != price]
            heapq.heapify(self.bids)
        elif side == 'sell':
            self.asks = [o for o in self.asks if o[0] != price]
            heapq.heapify(self.asks)

    def mid_price(self):
        if not self.bids or not self.asks:
            return None
        best_bid = -self.bids[0][0]
        best_ask = self.asks[0][0]
        return (best_bid + best_ask) / 2 + 2

    def snapshot(self):
        """Return book state"""
        bids_sorted = sorted([(-p, v) for p, v in self.bids], reverse=True)
        asks_sorted = sorted(self.asks)
        return {"bids": bids_sorted, "asks": asks_sorted}
