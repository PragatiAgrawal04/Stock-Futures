import requests
import pandas as pd
import numpy as np
import time
import xlwings as xw
from bs4 import BeautifulSoup
import datetime
import streamlit as st
import yfinance as yf
import csv

st.set_page_config(page_title="OPTSTK", layout="wide", initial_sidebar_state="collapsed")
st.sidebar.header("OC")
st.sidebar.write("Categorization of Call and Put options into 4 tabs: LB, SB, SC, LL")
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


# st.markdown("""
# <style>
#     .st-selectbox > option {
#         font-weight: bold;
#     }
# </style>
# """, unsafe_allow_html=True)
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

    atm_price = current_market_price(ticker)
    diffdata = pd.DataFrame({"Strike": strike, "Difference": [abs(x - atm_price) for x in strike]})
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

    return output_ce, output_pe, atm_price


def highlight_background(s):
    global ATM
    atm = ATM
    if s.STR_PRICE >= atm:
        col = ['background-color: white'] * 7 + ['background-color: antiquewhite'] * 6
    elif s.STR_PRICE < atm:
        col = ['background-color: antiquewhite'] * 6 + ['background-color: white'] * 7
    return col


def highlight_text(s):
    cat_ce = s.Category_CE
    cat_pe = s.Category_PE
    if cat_ce == 'LB' or cat_ce == 'SC':
        colce = 'green'
    else:
        colce = 'red'
    if cat_pe == 'LB' or cat_pe == 'SC':
        colpe = 'green'
    else:
        colpe = 'red'
    if colce == 'green' and colpe == 'green':
        col = ['color: black'] * 5 + ['color: green'] + ['color: black'] * 6 + ['color: green']
    elif colce == 'green' and colpe == 'red':
        col = ['color: black'] * 5 + ['color: green'] + ['color: black'] * 6 + ['color: red']
    elif colce == 'red' and colpe == 'green':
        col = ['color: black'] * 5 + ['color: red'] + ['color: black'] * 6 + ['color: green']
    else:
        col = ['color: black'] * 5 + ['color: red'] + ['color: black'] * 6 + ['color: red']
    return col


@st.experimental_fragment
def frag_table(table_number, selected_option='UBL'):
    global ATM
    global DATE_LIST
    global exp_date_list
    shares = pd.read_csv("FNO Stocks - All FO Stocks List, Technical Analysis Scanner.csv")
    share_list = list(shares["Symbol"])
    share_list.sort()
    selected_option = selected_option.strip()
    share_list.remove(selected_option)
    share_list = [selected_option] + share_list

    # exp_date_list_sel = DATE_LIST.copy()
    # exp_option = exp_date_list_sel[0]
    # print("LIST: ", exp_date_list_sel)
    # #exp_option = datetime.datetime.strptime(exp_option, "%d-%m-%Y").date().strftime('%d-%m-%Y')
    # print("EXP_OPTION:", exp_option)
    # exp_date_list_sel.remove(exp_option)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('##### Share List')
        selected_option = st.selectbox(label="", options=share_list, key="share_list" + str(table_number),
                                       label_visibility='collapsed')
        exp_date_list_sel = get_expiry_date_list(selected_option)
    with c2:
        st.markdown('##### Expiry List')
        exp_option = st.selectbox(label="", options=exp_date_list_sel, key="exp_list" + str(table_number),
                                  label_visibility='collapsed')
        if selected_option in share_list:
            ticker = selected_option
            output_ce, output_pe, atm = get_dataframe(ticker, exp_option)

    ATM = atm
    stock_ltp = current_market_price(ticker)
    low_52_week, high_52_week = fifty_two_week_high_low(ticker)

    keys = ['LB', 'SC', 'SB', 'LL']

    itm_ce_count = {key: output_ce.loc[:10]['Category'].tolist().count(key) for key in keys}
    otm_ce_count = {key: output_ce.loc[10:]['Category'].tolist().count(key) for key in keys}
    itm_pe_count = {key: output_pe.loc[10:]['Category'].tolist().count(key) for key in keys}
    otm_pe_count = {key: output_pe.loc[:10]['Category'].tolist().count(key) for key in keys}

    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        st.markdown('<h4 style="color: black;font-size: 20px;">CMP:  ' + str(stock_ltp)+"</h4>", unsafe_allow_html=True)
    with d2:
        st.markdown('<h4 style="color: black;font-size: 20px;">TIME:  ' +
                    datetime.datetime.now().strftime("%H:%M:%S")+"</h4>", unsafe_allow_html=True)
    with d3:
        st.markdown('<h4 style="color: red;font-size: 20px;">52 Week LOW:  '+str(low_52_week)+"</h4>", unsafe_allow_html=True)
    with d4:
        st.markdown('<h4 style="color: green;font-size: 20px;">52 Week HIGH:  ' + str(high_52_week) + "</h4>",
                    unsafe_allow_html=True)
    with d5:
        trend = st.empty()
        if (
                (
                        ((itm_pe_count['SB'] + itm_pe_count['LL']) > 5) and ((otm_pe_count['LL'] + otm_pe_count['SB']) > 5)
                ) and (
                        ((itm_ce_count['SC'] + itm_ce_count['LB']) >= 5) and ((otm_ce_count['LB'] + otm_ce_count['SC']) >= 5)
                    )):
            trend.markdown("""<h4 style="float: left;">
                            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TREND:
                            </h4>
                            <h4 style="float: right; 
                                       color: green;
                                       font-size: 30px;
                                       text-align:center;">
                            BULLISH
                            </h4>""", unsafe_allow_html=True)
        elif (
                (
                        ((itm_ce_count['SB'] + itm_ce_count['LL']) > 5) and ((otm_ce_count['LL'] + otm_ce_count['SB']) > 5)
                ) and (
                        ((itm_pe_count['SC'] + itm_pe_count['LB']) >= 5) and ((otm_pe_count['LB'] + otm_pe_count['SC']) >= 5)
                    )):
            trend.markdown("""<h4 style="float: left;">TREND:
                        </h4>
                        <h4 style="float: right; 
                                   color: red;
                                   font-size: 30px;
                                   text-align:center;">
                        BEARISH
                        </h4>""", unsafe_allow_html=True)
        elif otm_ce_count['SB'] >= 5 and otm_pe_count['SB'] >= 5:
            trend.markdown("""<h4 style="float: left;">
                            &nbsp;TREND:
                            </h4>
                            <h4 style="float: right; 
                                       color: blue;
                                       font-size: 30px;
                                       text-align:center;">
                            SIDEWAYS
                            </h4>""", unsafe_allow_html=True)
        else:
            trend.markdown("""<h4 style="float: left;">
                            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TREND:
                            </h4>
                            <h4 style="float: right; 
                                       color: black;
                                       font-size: 30px;
                                       text-align:center;">
                            NA
                            </h4>""", unsafe_allow_html=True)

    output_ce = output_ce[
        ['strikePrice', 'pchangeinOpenInterest', 'pChange', 'totalTradedVolume', 'impliedVolatility', 'lastPrice',
         'Category']]
    output_ce = output_ce.rename(columns={'strikePrice': 'STR_PRICE',
                                          'pchangeinOpenInterest': 'OI%_CE',
                                          'totalTradedVolume': 'Volume_CE',
                                          'impliedVolatility': 'IV_CE',
                                          'lastPrice': 'LTP_CE',
                                          'pChange': '%Change_CE',
                                          'Category': 'Category_CE'})
    output_pe = output_pe[
        ['strikePrice', 'pchangeinOpenInterest', 'pChange', 'totalTradedVolume', 'impliedVolatility', 'lastPrice',
         'Category']]
    output_pe = output_pe.rename(columns={'strikePrice': 'STR_PRICE',
                                          'pchangeinOpenInterest': 'OI%_PE',
                                          'totalTradedVolume': 'Volume_PE',
                                          'impliedVolatility': 'IV_PE',
                                          'lastPrice': 'LTP_PE',
                                          'pChange': '%Change_PE',
                                          'Category': 'Category_PE'})
    output = pd.merge(output_ce, output_pe, on='STR_PRICE')
    cols = output.columns.tolist()
    output = output[cols[1:7] + [cols[0]] + cols[7:]]
    output = output.style.apply(highlight_background, axis=1)
    output = output.apply(highlight_text, axis=1)
    output = output.set_properties(
        **{'text-align': 'center'}).set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'center')]}])
    output = output.format({'OI%_CE': "{:.2f}".format, '%Change_CE': "{:.1f}".format, 'LTP_CE': "{:.2f}".format,
                            'IV_CE': "{:.2f}".format, 'OI%_PE': "{:.2f}".format, '%Change_PE': "{:.1f}".format,
                            'LTP_PE': "{:.2f}".format, 'IV_PE': "{:.2f}".format, 'STR_PRICE': "{:.1f}".format})
    st.markdown("""<style>
                    .col_heading
                    {text-align: center;}
                    </style>""", unsafe_allow_html=True)
    output.columns = ['<div class="col_heading">' + col + '</div>' for col in output.columns]

    st.write(output.to_html(escape=False), unsafe_allow_html=True)

    if ('share_list2' in st.session_state) and ('share_list3' in st.session_state):
        curr = pd.DataFrame({'table1': [st.session_state["share_list1"]],
                             'exp1': [st.session_state["exp_list1"]],
                             'table2': [st.session_state["share_list2"]],
                             'exp2': [st.session_state["exp_list2"]],
                             'table3': [st.session_state["share_list3"]],
                             'exp3': [st.session_state["exp_list3"]],
                             'timestamp': [datetime.datetime.now()]
                             })
        if len(hist_df) > 30:
            curr.to_csv('history.csv', mode='w', index=False, header=True)
        else:
            curr.to_csv('history.csv', mode='a', index=False, header=False)
    st.write("---")


#########################################################################################################
st.markdown('## OPTION CHAIN ANALYSIS')
hist = pd.read_csv("history.csv")
hist_df = pd.DataFrame(hist)

print(len(hist_df))

if len(hist_df) > 0:
    last_rec = hist_df.tail(1)
    print(last_rec)
    frag_table(1, last_rec['table1'].item())
    frag_table(2, last_rec['table2'].item())
    frag_table(3, last_rec['table3'].item())
else:
    frag_table(1, 'RELIANCE')
    frag_table(2, 'VEDL')
    frag_table(3, 'INFY')
