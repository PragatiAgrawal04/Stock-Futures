import requests
import pandas as pd
import numpy as np
import time
import xlwings as xw
from bs4 import BeautifulSoup
import datetime
import streamlit as st
import pytz
import yfinance as yf
import csv

st.set_page_config(page_title="Dashboard", layout="wide", initial_sidebar_state="collapsed")
st.sidebar.header("Dashboard")
exchange = "NSE"
ATM = 0
symbols = pd.read_csv('symbols.csv')
stk_symbol_list = list(symbols['stock_symbols'])
yf_stock_symbol_list = list(symbols['stk_symbol_yf'])
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }

        </style>
        """, unsafe_allow_html=True)


def get_expiry_date_list(symb):
    year = datetime.datetime.now().year
    curr_month = datetime.datetime.now().month
    curr_day = datetime.datetime.now().day
    exp = []
    for month in range(curr_month, 13):
        if month == curr_month:
            day = curr_day
        else:
            day = 1
        if month <= 9:
            date = f"{year}-0{month}-01"
        else:
            date = f"{year}-{month}-01"
        df_month = pd.to_datetime(date)
        if symb in ['BANKNIFTY', 'NIFTY', 'FINNIFTY']:
            if symb == 'BANKNIFTY':
                last_date = (df_month + pd.tseries.offsets.MonthEnd(1)).day
                day_ind = 2
            elif symb == 'NIFTY':
                last_date = (df_month + pd.tseries.offsets.MonthEnd(1)).day
                day_ind = 3
            elif symb == 'FINNIFTY':
                last_date = (df_month + pd.tseries.offsets.MonthEnd(1)).day
                day_ind = 1
            for day in range(day, last_date):
                temp_date = datetime.date(year, month, day)
                if temp_date.weekday() == day_ind:
                    exp.append(temp_date)
                else:
                    continue
        else:
            last_date = df_month + pd.tseries.offsets.MonthEnd(1)
            offset = (last_date.weekday() - 3) % 7
            df_expiry = last_date - pd.to_timedelta(offset, unit='D')
            exp.append(df_expiry.date())

    date_list = []
    today = datetime.date.today()
    for i in range(len(exp)):
        x = (exp[i] - today).days
        if x >= 0:
            date_list.append(exp[i].strftime('%d-%m-%Y'))
    return date_list


def nifty_cash(date, symbol):
    while date.weekday() == 5 or date.weekday() == 6:
            date = date - datetime.timedelta(1)
    data = yf.download(symbol, start=date, end=date + datetime.timedelta(1), interval='1m')
    data = pd.DataFrame(data)
    data['Date'] = [i.date() for i in data.index]
    data['Time'] = [i.time() for i in data.index]
    data = data[['Date', 'Time', 'Open', 'High', 'Low', 'Close']].reset_index(drop=True)
    return data


def current_market_price(ticker):
    global yf_stock_symbol_list
    global stk_symbol_list
    yfsymb = (yf_stock_symbol_list[stk_symbol_list.index(ticker)])
    chk_date = datetime.date.today()
    nifty_cash_data = nifty_cash(chk_date, yfsymb)
    current_price = list(nifty_cash_data['Close'])[-1]
    return round(current_price, 2)


def fifty_two_week_high_low(ticker):
    global yf_stock_symbol_list
    global stk_symbol_list
    yfsymb = (yf_stock_symbol_list[stk_symbol_list.index(ticker)])
    data = yf.download(yfsymb, period="1y", auto_adjust=True, prepost=True, threads=True)
    low_52_week = round(float(data['Low'].min()), 2)
    high_52_week = round(float(data['High'].max()), 2)
    return low_52_week, high_52_week


def categorize(data):
    d = data.copy(deep=True)
    cat = []
    oich = d['pchangeinOpenInterest']
    pch = d['pChange']
    for ind in range(len(d)):
        if pch[ind] > 0 and oich[ind] > 0:
            cat.append("LB")
        elif pch[ind] < 0 < oich[ind]:
            cat.append("SB")
        elif pch[ind] > 0 > oich[ind]:
            cat.append("SC")
        else:
            cat.append("LL")
    d['Category'] = cat
    return d


def get_dataframe(ticker, exp_date_selected):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}

    main_url = "https://www.nseindia.com/"
    response = requests.get(main_url, headers=headers)
    cookies = response.cookies
    if ticker in ['BANKNIFTY', 'NIFTY', 'FINNIFTY']:
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={ticker}"
    else:
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={ticker}"
    option_chain_data = requests.get(url, headers=headers, cookies=cookies)

    data = option_chain_data.json()["records"]["data"]
    ocdata = []

    for i in data:
        for j, k in i.items():
            if j == "CE" or j == "PE":
                info = k
                info["instrumentType"] = j
                ocdata.append(info)

    df = pd.DataFrame(ocdata)

    strike = df.strikePrice.unique().tolist()
    strike_size = int(strike[int(len(strike) / 2) + 1]) - int(strike[int(len(strike) / 2)])

    cmp = current_market_price(ticker)
    diffdata = pd.DataFrame({"Strike": strike, "Difference": [abs(x - cmp) for x in strike]})
    atm_price = diffdata.loc[diffdata['Difference'] == diffdata['Difference'].min()]['Strike'].head(1).item()

    output_ce = pd.DataFrame()
    output_pe = pd.DataFrame()

    fd = df.copy(deep=True)
    fd_pe = df.copy(deep=True)

    # (for ce)convert expiry date in particular format
    fd = fd.reset_index(drop=True)
    for i in range(len(fd)):
        expiry_date_str = fd["expiryDate"].iloc[i]
        temp_expiry = datetime.datetime.strptime(expiry_date_str, '%d-%b-%Y')
        result_expiry = temp_expiry.strftime('%d-%m-%Y')
        fd.at[i, "expiryDate"] = result_expiry
    # print(fd)
    # print(type(fd["expiryDate"].iloc[0]))

    # (for pe) convert expiry date in particular format
    fd_pe = fd_pe.reset_index(drop=True)
    for i in range(len(fd_pe)):
        expiry_date_str_pe = fd_pe["expiryDate"].iloc[i]
        temp_expiry_pe = datetime.datetime.strptime(expiry_date_str_pe, '%d-%b-%Y')
        result_expiry_pe = temp_expiry_pe.strftime('%d-%m-%Y')
        fd_pe.at[i, "expiryDate"] = result_expiry_pe

    adjusted_expiry = exp_date_selected
    adjusted_expiry_pe = exp_date_selected

    # (subset_ce (CE))
    subset_ce = fd[(fd.instrumentType == "CE") & (fd.expiryDate == adjusted_expiry)].reset_index(drop=True)
    ind_atm_ce = subset_ce[(subset_ce.strikePrice == atm_price)].index.tolist()[0]
    subset_ce = subset_ce.loc[ind_atm_ce - 10:ind_atm_ce + 10, ].reset_index(drop=True)
    output_ce = pd.concat([output_ce, subset_ce]).reset_index(drop=True)
    output_ce = categorize(output_ce)

    # (subset_pe (PE))
    subset_pe = fd_pe[(fd_pe.instrumentType == "PE") & (fd_pe.expiryDate == adjusted_expiry_pe)].reset_index(drop=True)
    ind_atm_pe = subset_pe[(subset_pe.strikePrice == atm_price)].index.tolist()[0]
    subset_pe = subset_pe.loc[ind_atm_pe - 10:ind_atm_pe + 10, ].reset_index(drop=True)
    output_pe = pd.concat([output_pe, subset_pe]).reset_index(drop=True)
    output_pe = categorize(output_pe)

    output_ce.reset_index(drop=True, inplace=True)
    output_pe.reset_index(drop=True, inplace=True)

    return output_ce, output_pe, cmp


def highlight_background(s):
    if s.Trend == 'NA':
        return ['background-color: white'] * len(s)
    elif s.Trend == 'BULLISH':
        return ['background-color: palegreen'] * len(s)
    elif s.Trend == 'BEARISH':
        return ['background-color: lightcoral'] * len(s)
    elif s.Trend == 'SIDEWAYS':
        return ['background-color: cornflowerblue'] * len(s)
#######################################################################################################################
st.markdown("## OC Top 20 Trend Dashboard")
st.markdown('<h4 style="color: black;font-size: 20px;">TIME:  ' +
                    datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S")+"</h4>", unsafe_allow_html=True)

shares = pd.read_csv("ind_nifty50list.csv")
share_list = list(shares["Symbol"])
dashboard_df = pd.DataFrame()
dashboard = st.empty()
for symbol in share_list:
    print(symbol)
    trend = ''
    exp_date_list_sel = get_expiry_date_list(symbol)
    latest_expiry = exp_date_list_sel[0]
    ticker = symbol
    output_ce, output_pe, stock_ltp = get_dataframe(ticker, latest_expiry)
    low_52_week, high_52_week = fifty_two_week_high_low(ticker)

    keys = ['LB', 'SC', 'SB', 'LL']

    itm_ce_count = {key: output_ce.loc[:10]['Category'].tolist().count(key) for key in keys}
    otm_ce_count = {key: output_ce.loc[10:]['Category'].tolist().count(key) for key in keys}
    itm_pe_count = {key: output_pe.loc[10:]['Category'].tolist().count(key) for key in keys}
    otm_pe_count = {key: output_pe.loc[:10]['Category'].tolist().count(key) for key in keys}

    if (
            (
                    ((itm_pe_count['SB'] + itm_pe_count['LL']) > 5) and ((otm_pe_count['LL'] + otm_pe_count['SB']) > 5)
            ) and (
            ((itm_ce_count['SC'] + itm_ce_count['LB']) >= 5) and ((otm_ce_count['LB'] + otm_ce_count['SC']) >= 5)
                    )):
        trend = 'BULLISH'
    elif(
                (
                        ((itm_ce_count['SB'] + itm_ce_count['LL']) > 5) and ((otm_ce_count['LL'] + otm_ce_count['SB']) > 5)
                ) and (
                        ((itm_pe_count['SC'] + itm_pe_count['LB']) >= 5) and ((otm_pe_count['LB'] + otm_pe_count['SC']) >= 5)
                    )):
        trend = 'BEARISH'
    elif otm_ce_count['SB'] >= 5 and otm_pe_count['SB'] >= 5:
        trend = 'SIDEWAYS'
    else:
        trend = 'NA'

    current_symbol_data = pd.DataFrame({'Symbol': [symbol],
                                        'Latest Expiry': [latest_expiry],
                                        'CMP': [stock_ltp],
                                        '52 week LOW': [low_52_week],
                                        '52 week HIGH': [high_52_week],
                                        'Trend': [trend]
                                        })
    dashboard_df = pd.concat([dashboard_df, current_symbol_data], axis=0).reset_index(drop=True)
    dashboard_df = dashboard_df.style.apply(highlight_background, axis=1)
    dashboard_df = dashboard_df.set_properties(
        **{'text-align': 'center'}).set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'center')]}])
    dashboard_df = dashboard_df.format({'52 week LOW': "{:.2f}".format, '52 week HIGH': "{:.2f}".format, 'CMP': "{:.2f}".format})
    st.markdown("""<style>
                    .col_heading
                    {text-align: center;}
                    </style>""", unsafe_allow_html=True)
    dashboard_df.columns = ['<div class="col_heading">' + col + '</div>' for col in dashboard_df.columns]
    dashboard.write(dashboard_df.to_html(escape=False), unsafe_allow_html=True)
    dashboard_df = dashboard_df.data
dashboard_df.to_csv('dashboard.csv', mode='w', index=False, header=True)
