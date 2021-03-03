#!/usr/bin/env python
# -*- coding: utf-8 -*-

from example.market.HuobiDMService import HuobiDM

from huobi.client.market import MarketClient
from huobi.utils import *

import time
import datetime

contractlist = ['btc', 'eth', 'link', 'dot', 'eos', 'trx', 'ada', 'ltc', 'bch', 'xrp', 'bsv', 'etc', 'fil']

watch_thresh = 10

# Futures setup
URL = 'https://api.hbdm.com'

ACCESS_KEY = ''
SECRET_KEY = ''

dm = HuobiDM(URL, ACCESS_KEY, SECRET_KEY)

# BB setup
market_client = MarketClient()

while True:
    for contract in contractlist:
        # Get BB MarketData
        bbsymbol = contract.lower() + 'usdt'
        depth = market_client.get_pricedepth(bbsymbol, DepthStep.STEP0, 20)
        bb_ask = depth.asks[19].price
        # Get Future MarketData
        futsymbol = contract.upper() + '_NQ'
        delivery_date = dm.get_contract_info(symbol=contract.upper(), contract_type="next_quarter")['data'][0]['delivery_date']
        waiteday = (datetime.datetime.strptime(delivery_date, '%Y%m%d') - datetime.datetime.now()).days + 2
        fut_depth = dm.get_contract_depth(symbol=futsymbol, type='step6')
        fut_bid = fut_depth['tick']['bids'][19][0]
        jc = round((fut_bid - bb_ask) / bb_ask * 100 * 365 / waiteday, 6)
        if jc > watch_thresh:
            print(contract + ' NQ: ' + str(jc))
        time.sleep(1)

        futsymbol = contract.upper() + '_CQ'
        delivery_date = dm.get_contract_info(symbol=contract.upper(), contract_type="quarter")['data'][0]['delivery_date']
        waiteday = (datetime.datetime.strptime(delivery_date, '%Y%m%d') - datetime.datetime.now()).days + 2
        fut_depth = dm.get_contract_depth(symbol=futsymbol, type='step6')
        fut_bid = fut_depth['tick']['bids'][19][0]
        jc = round((fut_bid - bb_ask) / bb_ask * 100 * 365 / waiteday, 6)
        if jc > watch_thresh:
            print(contract + ' CQ: ' + str(jc))
        time.sleep(1)
    print("******************************************")
    time.sleep(30)
