function LeeksReaper() {
    var self = {}
    self.numTick = 0
    self.lastTradeId = 0
    self.vol = 0
    self.askPrice = 0
    self.bidPrice = 0
    self.orderBook = {Asks:[], Bids:[]}
    self.prices = []
    self.tradeOrderId = 0
    self.p = 0.5
    self.account = null
    self.preCalc = 0
    self.preNet = 0

    self.updateTrades = function() {
        var trades = _C(exchange.GetTrades)
        if (self.prices.length == 0) {
            while (trades.length == 0) {
                trades = trades.concat(_C(exchange.GetTrades))
            }
            for (var i = 0; i < 15; i++) {
                self.prices[i] = trades[trades.length - 1].Price
            }
        }
        self.vol = 0.7 * self.vol + 0.3 * _.reduce(trades, function(mem, trade) {
            // Huobi not support trade.Id
            if ((trade.Id > self.lastTradeId) || (trade.Id == 0 && trade.Time > self.lastTradeId)) {
                self.lastTradeId = Math.max(trade.Id == 0 ? trade.Time : trade.Id, self.lastTradeId)
                mem += trade.Amount
            }
            return mem
        }, 0)

    }
    self.updateOrderBook = function() {
        var orderBook = _C(exchange.GetDepth)
        self.orderBook = orderBook
        if (orderBook.Bids.length < 3 || orderBook.Asks.length < 3) {
            return
        }
        self.bidPrice = orderBook.Bids[0].Price * 0.618 + orderBook.Asks[0].Price * 0.382 + 0.01
        self.askPrice = orderBook.Bids[0].Price * 0.382 + orderBook.Asks[0].Price * 0.618 - 0.01
        self.prices.shift()
        self.prices.push(_N((orderBook.Bids[0].Price + orderBook.Asks[0].Price) * 0.35 +
            (orderBook.Bids[1].Price + orderBook.Asks[1].Price) * 0.1 +
            (orderBook.Bids[2].Price + orderBook.Asks[2].Price) * 0.05))
    }
    self.balanceAccount = function() {
        var account = exchange.GetAccount()
        if (!account) {
            return
        }
        self.account = account
        var now = new Date().getTime()
        if (self.orderBook.Bids.length > 0 && now - self.preCalc > (CalcNetInterval * 1000)) {
            self.preCalc = now
            var net = _N(account.Balance + account.FrozenBalance + self.orderBook.Bids[0].Price * (account.Stocks + account.FrozenStocks))
            if (net != self.preNet) {
                self.preNet = net
                LogProfit(net)
            }
        }
        self.btc = account.Stocks
        self.cny = account.Balance
        self.p = self.btc * self.prices[self.prices.length-1] / (self.btc * self.prices[self.prices.length-1] + self.cny)
        var balanced = false
        
        if (self.p < 0.48) {
            Log("开始平衡", self.p)
            self.cny -= 300
            if (self.orderBook.Bids.length >0) {
                exchange.Buy(self.orderBook.Bids[0].Price + 0.00, 0.01)
                exchange.Buy(self.orderBook.Bids[0].Price + 0.01, 0.01)
                exchange.Buy(self.orderBook.Bids[0].Price + 0.02, 0.01)
            }
        } else if (self.p > 0.52) {
            Log("开始平衡", self.p)
            self.btc -= 0.03
            if (self.orderBook.Asks.length >0) {
                exchange.Sell(self.orderBook.Asks[0].Price - 0.00, 0.01)
                exchange.Sell(self.orderBook.Asks[0].Price - 0.01, 0.01)
                exchange.Sell(self.orderBook.Asks[0].Price - 0.02, 0.01)
            }
        }
        Sleep(BalanceTimeout)
        var orders = exchange.GetOrders()
        if (orders) {
            for (var i = 0; i < orders.length; i++) {
                if (orders[i].Id != self.tradeOrderId) {
                    exchange.CancelOrder(orders[i].Id)
                }
            }
        }
    }

    self.poll = function() {
        self.numTick++
        self.updateTrades()
        self.updateOrderBook()
        self.balanceAccount()
        
        var burstPrice = self.prices[self.prices.length-1] * BurstThresholdPct
        var bull = false
        var bear = false
        var tradeAmount = 0
        if (self.account) {
            LogStatus(self.account, 'Tick:', self.numTick, ', lastPrice:', self.prices[self.prices.length-1], ', burstPrice: ', burstPrice)
        }
        
        if (self.numTick > 2 && (
            self.prices[self.prices.length-1] - _.max(self.prices.slice(-6, -1)) > burstPrice ||
            self.prices[self.prices.length-1] - _.max(self.prices.slice(-6, -2)) > burstPrice && self.prices[self.prices.length-1] > self.prices[self.prices.length-2]
            )) {
            bull = true
            tradeAmount = self.cny / self.bidPrice * 0.99
        } else if (self.numTick > 2 && (
            self.prices[self.prices.length-1] - _.min(self.prices.slice(-6, -1)) < -burstPrice ||
            self.prices[self.prices.length-1] - _.min(self.prices.slice(-6, -2)) < -burstPrice && self.prices[self.prices.length-1] < self.prices[self.prices.length-2]
            )) {
            bear = true
            tradeAmount = self.btc
        }
        if (self.vol < BurstThresholdVol) {
            tradeAmount *= self.vol / BurstThresholdVol
        }
        
        if (self.numTick < 5) {
            tradeAmount *= 0.8
        }
        
        if (self.numTick < 10) {
            tradeAmount *= 0.8
        }
        
        if ((!bull && !bear) || tradeAmount < MinStock) {
            return
        }
        var tradePrice = bull ? self.bidPrice : self.askPrice
        while (tradeAmount >= MinStock) {
            var orderId = bull ? exchange.Buy(self.bidPrice, tradeAmount) : exchange.Sell(self.askPrice, tradeAmount)
            Sleep(200)
            if (orderId) {
                self.tradeOrderId = orderId
                var order = null
                while (true) {
                    order = exchange.GetOrder(orderId)
                    if (order) {
                        if (order.Status == ORDER_STATE_PENDING) {
                            exchange.CancelOrder(orderId)
                            Sleep(200)
                        } else {
                            break
                        }
                    }
                }
                self.tradeOrderId = 0
                tradeAmount -= order.DealAmount
                tradeAmount *= 0.9
                if (order.Status == ORDER_STATE_CANCELED) {
                    self.updateOrderBook()
                    while (bull && self.bidPrice - tradePrice > 0.1) {
                        tradeAmount *= 0.99
                        tradePrice += 0.1
                    }
                    while (bear && self.askPrice - tradePrice < -0.1) {
                        tradeAmount *= 0.99
                        tradePrice -= 0.1
                    }
                }
            }
        }
        self.numTick = 0
    }
    return self
}

function main() {
    var reaper = LeeksReaper()
    while (true) {
        reaper.poll()
        Sleep(TickInterval)
    }
}