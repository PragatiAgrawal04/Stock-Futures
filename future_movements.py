import requests
import zipfile
import io
import pandas as pd
from datetime import date, timedelta
import datetime
import numpy as np
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import math
import time

st.set_page_config(
    page_title="FUTSTK",
    layout="wide",  # Use the entire screen width
    initial_sidebar_state="collapsed",  # Initially hide the sidebar
)
st.sidebar.header("Future Movements")
holidays = [date(2024, 1, 22), date(2024, 1, 26), date(2024, 3, 8), date(2024, 3, 25),
            date(2024, 3, 29), date(2024, 4, 11), date(2024, 4, 17), date(2024, 5, 1),
            date(2024, 5, 20), date(2024, 6, 17), date(2024, 7, 17), date(2024, 8, 15),
            date(2024, 9, 2), date(2024, 11, 1), date(2024, 11, 15), date(2024, 12, 25)]
option = 'FUTSTK'
symbols = pd.read_csv('symbols.csv')
stk_symbol_list = list(symbols['stock_symbols'])
yf_stock_symbol_list = list(symbols['stk_symbol_yf'])


def download_extract_zip(url, headers):
    response = requests.get(url, headers=headers)
    with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
        for zipinfo in thezip.infolist():
            with thezip.open(zipinfo) as thefile:
                yield zipinfo.filename, thefile


headers = {
    'accept': 'application/json, text/javascript, */*; q=0.01',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.9',
    'referer': 'https://www.nseindia.com/api/chart-databyindex?index=OPTIDXBANKNIFTY25-01-2024CE46000.00',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'x-requested-with': 'XMLHttpRequest',
}


def read_bhavcopy_data(data, o):
    data = data.rename(columns=lambda x: x.strip())
    data['TIMESTAMP'] = [datetime.datetime.strptime(i, "%d-%b-%Y").date() for i in data['TIMESTAMP']]
    data['EXPIRY_DT'] = [datetime.datetime.strptime(i, "%d-%b-%Y").date() for i in data['EXPIRY_DT']]
    # data['Underlying Value']=[float(list(data['Underlying Value'])[i]) if list(data['Underlying Value'])[i]!='-' else np.nan for i in range(0,len(data))]
    data = data.sort_values(by=['EXPIRY_DT', 'TIMESTAMP'], ignore_index=True)
    data = data.drop(['Unnamed: 15'], axis=1)
    data = data.loc[(data['INSTRUMENT'] == o)]
    return data


def extract_bhavcopy(opt):
    entire_data = pd.DataFrame()
    time_now = time.time()

    date_today = date.today()
    dates = [date_today - timedelta(i) for i in range(1, 6)]

    for i in dates:
        if (i.strftime("%A") in ['Saturday', 'Sunday']) or (i in holidays):
            continue
        else:

            dd = i.day
            mm = i.strftime("%B")[:3].upper()
            yy = str(i.year)

            url = "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/" + yy + "/" + mm + "/fo" + f"{dd:02d}" + mm + yy + "bhav.csv.zip"
            for filename, file in download_extract_zip(url, headers):
                if filename.endswith('.csv'):
                    df = pd.read_csv(file)
                    df = read_bhavcopy_data(df, opt)
                entire_data = pd.concat([entire_data, df], axis=0, ignore_index=True)
    entire_data = entire_data.sort_values(by=['EXPIRY_DT', 'TIMESTAMP'], ignore_index=True)
    final_data = pd.DataFrame()
    for ind, data in entire_data.iterrows():
        if abs((data['TIMESTAMP'] - data['EXPIRY_DT']).days) > 2:
            final_data = pd.concat([final_data, data], axis=1)
    entire_data = final_data.transpose().reset_index(drop=True)
    return entire_data


def action_setting(futdata):
    exp = futdata['EXPIRY_DT'].unique()
    fdata = pd.DataFrame()
    for i in exp:
        edata = futdata[futdata['EXPIRY_DT'] == i].reset_index(drop=True)
        cat, act, spchange, oichange = [np.nan], [np.nan], [np.nan], [np.nan]
        for j in range(1, len(edata)):
            asp, bsp = edata['SETTLE_PR'][j - 1], edata['SETTLE_PR'][j]
            spchange.append(np.round(((bsp / asp) - 1) * 100, 2))
            oichange.append(np.round((edata['CHG_IN_OI'][j] / edata['OPEN_INT'][j]) * 100, 2))
            sp = [0 if spchange[j] < 0 else 1]
            oi = [0 if oichange[j] < 0 else 1]
            if sp == [1] and oi == [1]:
                cat.append("LB")
                act.append("Buy")
            elif sp == [0] and oi == [1]:
                cat.append("SB")
                act.append("Sell")
            elif sp == [1] and oi == [0]:
                cat.append("SC")
                act.append("Buy")
            elif sp == [0] and oi == [0]:
                cat.append("LL")
                act.append("Sell")
        edata['Settle Price Change'] = spchange
        edata['Movement of OI'] = oichange
        edata['Category'] = cat
        edata['Action'] = act
        fdata = pd.concat([fdata, edata], axis=0)
        fdata = fdata.set_index([pd.Index(range(0, len(fdata)))])
    # fdata=fdata.sort_values(by=['EXPIRY_DT','TIMESTAMP'],ignore_index=True)
    return fdata


def nifty_cash(date, symbol):
    data = yf.download(symbol, start=date, end=date + timedelta(1), interval='5m')
    data = pd.DataFrame(data)
    # data['DateTime']=data.index
    data['Date'] = [i.date() for i in data.index]
    data['Time'] = [i.time() for i in data.index]
    # data=data.drop(['DateTime','Volume'],axis=1)
    # change_in_close=[np.round(data.iloc[0,3]-actfutdata.iloc[-1,2],2)]
    # for i in range(1,len(data)):
    #     change_in_close.append(np.round(data.iloc[i,3]-data.iloc[i-1,3],2))
    # data['Change_in_close']=change_in_close
    # data=data.reset_index(drop=True)
    data = data[['Date', 'Time', 'Open', 'High', 'Low', 'Close']].reset_index(drop=True)
    return data


def chk_pnl_stock(actfutdata, holder):
    symb = actfutdata['SYMBOL'].unique().tolist()
    x = []
    y = []
    current_price = []
    chk_date = date.today()
    nifty_cash_all_data = pd.DataFrame()
    if (chk_date.strftime("%A") in ['Saturday', 'Sunday']) or (chk_date in holidays):
        nifty_cash_data = pd.DataFrame()
        st.write("MARKET IS CLOSED")
    else:
        for i in symb:
            yfsymb = (yf_stock_symbol_list[stk_symbol_list.index(i)])
            test_date = date.today() - timedelta(1)
            while (test_date.strftime("%A") in ['Saturday', 'Sunday']) or (test_date in holidays):
                test_date = test_date - timedelta(1)
            nifty_cash_data = nifty_cash(chk_date, yfsymb)
            yest_closing = actfutdata.loc[(actfutdata['TIMESTAMP'] == test_date) &
                                          (actfutdata['EXPIRY_DT'] == actfutdata['EXPIRY_DT'].unique()[0]) &
                                          (actfutdata['SYMBOL'] == i)]['CLOSE'].item()
            action = actfutdata.loc[(actfutdata['TIMESTAMP'] == test_date) &
                                    (actfutdata['EXPIRY_DT'] == actfutdata['EXPIRY_DT'].unique()[0]) &
                                    (actfutdata['SYMBOL'] == i)]['Action'].item()
            if action == 'Buy':
                nifty_cash_data['BUY PnL'] = nifty_cash_data['Close'] - yest_closing
                yitem = list(nifty_cash_data['BUY PnL'])
                yitem = np.array(yitem)
                y.append(yitem)
            elif action == 'Sell':
                nifty_cash_data['SELL PnL'] = yest_closing - nifty_cash_data['Close']
                yitem = list(nifty_cash_data['SELL PnL'])
                yitem = np.array(yitem)
                y.append(yitem)
            x.append([str(i.hour) + ":" + str(i.minute) for i in nifty_cash_data['Time']])
            current_price.append(list(nifty_cash_data['Close'])[-1])
            nifty_cash_data = pd.concat([pd.DataFrame({"Symbol": [i] * len(nifty_cash_data)}), nifty_cash_data], axis=1)
            nifty_cash_all_data = pd.concat([nifty_cash_all_data, nifty_cash_data], axis=0).reset_index(drop=True)
    movement = [0]
    walk_list = [0]
    for ind in range(1, len(nifty_cash_all_data)):
        curr_open = nifty_cash_all_data['Open'].tolist()[ind]
        prev_close = nifty_cash_all_data['Close'].tolist()[ind-1]
        diff = curr_open - prev_close
        walk = movement[ind-1] + (diff)
        walk_list.append(walk)
        movement.append(diff)
    nifty_cash_all_data['Movement'] = movement
    nifty_cash_all_data['Walk'] = walk_list
        # print(x)
        # print(y)
    frag_plots(symb, x, y, current_price, holder)

    return nifty_cash_all_data


@st.experimental_fragment
def frag(box_num, action_data):
    def nextpage(men_opt):
        if "menu" not in st.session_state:
            st.session_state.menu = 'All'
        else:
            st.session_state.menu = men_opt


    menu_option_list = ['All', 'Top Movements']
    menu_option_list = menu_option_list + stk_symbol_list
    menu_option = st.selectbox("Select Category", menu_option_list, key='menu_option')
    nextpage(menu_option)
    data_action = st.empty()
    plot_placeholder = st.empty()
    move_plot_placeholder = st.empty()
    movement_data = st.empty()
    if st.session_state.menu == 'All':
        action_data = action_data.loc[
            action_data['TIMESTAMP'] == list(action_data['TIMESTAMP'].unique())[-1]].reset_index(drop=True)
        data_action.dataframe(action_data[['INSTRUMENT', 'SYMBOL', 'EXPIRY_DT', 'CLOSE', 'SETTLE_PR', 'TIMESTAMP',
                                           'Settle Price Change', 'Movement of OI', 'Category', 'Action']],
                              use_container_width=True)
    elif st.session_state.menu == 'Top Movements':
        action_data = action_data.loc[
            action_data['TIMESTAMP'] == list(action_data['TIMESTAMP'].unique())[-1]].reset_index(drop=True)
        action_data = action_data.loc[
            (np.abs(action_data['Settle Price Change']) >= 2) &
            (np.abs(action_data['Movement of OI']) >= 8)].reset_index(drop=True)
        data_action.dataframe(action_data[['INSTRUMENT', 'SYMBOL', 'EXPIRY_DT', 'CLOSE', 'SETTLE_PR', 'TIMESTAMP',
                                           'Settle Price Change', 'Movement of OI', 'Category', 'Action']],
                              use_container_width=True)
        sym = ['All'] + list(action_data['SYMBOL'].unique())

        def nextstock(men_opt):
            if "top_menu" not in st.session_state:
                st.session_state.top_menu = 'All'
            else:
                st.session_state.top_menu = men_opt

        top_sel = plot_placeholder.selectbox("Select STOCK", sym)
        nextstock(top_sel)
        if st.session_state.top_menu == 'All':
            sym.remove('All')
            act_stk_all = pd.DataFrame()
            for i in range(0, len(sym)):
                act_stk_data = action_data.loc[action_data['SYMBOL'] == sym[i]].reset_index(drop=True).tail(1)
                act_stk_all = pd.concat([act_stk_all, act_stk_data], axis=0).reset_index(drop=True)
            live_data = chk_pnl_stock(act_stk_all, move_plot_placeholder)
        elif st.session_state.top_menu != 'All':
            act_stk_data = action_data.loc[action_data['SYMBOL'] == top_sel].reset_index(drop=True).tail(1)
            plot1, plot2 = move_plot_placeholder.columns(2)
            live_data = chk_pnl_stock(act_stk_data, plot1)
            movement_plot(plot2, live_data)

    elif not (st.session_state.menu in ['All', 'Top Movements']):
        action_data = action_data.loc[action_data['SYMBOL'] == menu_option].reset_index(drop=True)
        data_action.dataframe(action_data[['INSTRUMENT', 'SYMBOL', 'EXPIRY_DT', 'CLOSE', 'SETTLE_PR', 'TIMESTAMP',
                                           'Settle Price Change', 'Movement of OI', 'Category', 'Action']],
                              use_container_width=True)
        # sym = list(action_data['SYMBOL'].unique())[0]
        plot1, plot2 = plot_placeholder.columns(2)
        live_data = chk_pnl_stock(action_data.tail(1), plot1)
        movement_plot(plot2, live_data)



@st.experimental_fragment
def frag_plots(symb, data_x, data_y, cp, hold):
    if st.session_state.top_menu == 'All' and st.session_state.menu == 'Top Movements':
        x = math.ceil(len(symb) / 2)
        fig, ax = plt.subplots(nrows=x, ncols=2, figsize=(8, 10))
        plot = 0
        for a in range(x):
            for b in range(2):
                print("Plot:", plot, "    ", a, b)
                ax[a, b].set_title(symb[plot] + "| CP:" + str(np.round(cp[plot], 2)), fontsize=6)
                labs = [data_x[plot][k] for k in range(0, len(data_x[plot]), 3)]
                xt = [data_x[plot].index(l) for l in labs]
                print(labs)
                ax[a, b].set_xticks(ticks=xt,
                                    labels=labs,
                                    rotation=90, fontsize=4)
                num = 10
                if int(min(data_y[plot])) - int(max(data_y[plot])) < 10:
                    num = 5
                yt = np.round(np.linspace(int(min(data_y[plot])), int(max(data_y[plot])), num), 2)
                print(yt)
                # yt = [np.linspace(-5,5,10)]
                ax[a, b].set_yticks(ticks=yt, labels=yt, fontsize=4)
                ax[a, b].set_ylim(min(-1, min(data_y[plot]) - 2), max(1, max(data_y[plot]) + 2))
                ax[a, b].set_ylabel("Profit", fontsize=6)
                ax[a, b].grid()
                ax[a, b].axhline(color="red", linewidth=0.8)
                ax[a, b].plot(data_y[plot], linewidth = 0.5)
                plot += 1
                print("After updation:", (a + 1) * (b + 1), "      ", plot)
                if plot == len(symb):
                    if len(symb) == (a + 1) * (b + 1):
                        continue
                    else:
                        ax[a, b + 1].set_visible(False)
                        break
        plt.subplots_adjust(hspace=1.5)
        hold.pyplot(fig, use_container_width=False)
    elif not (st.session_state.menu in ['All', 'Top Movements']) or st.session_state.top_menu != 'All':
        # st.write('<style>.center {justify-content: center;align-items: center}</style><div class="center">Chart1', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.title(symb[0] + "| CP:" + str(np.round(cp[0], 2)))
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(fontsize=6)
        plt.ylim(min(-1, min(data_y[0]) - 2), max(1, max(data_y[0]) + 2))
        plt.grid()
        plt.axhline(color="black")
        plt.plot(data_x[0], data_y[0])
        hold.pyplot(fig, use_container_width=True)

    # st.write('<div class="center">Chart 1', unsafe_allow_html=True)
    # st.write('</div>', unsafe_allow_html=True)


def movement_plot(hold, data):
    fig, ax = plt.subplots(figsize=(10, 5))
    symb = data['Symbol'].unique().tolist()
    data_y = data['Walk']
    data_x = [str(i.hour) + ":" + str(i.minute) for i in data['Time']]
    plt.title(symb[0] + "| Movement")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.ylim(min(-1, min(data_y) - 0.02*max(data_y)), max(1, max(data_y) + 0.02*max(data_y)))
    plt.grid()
    plt.axhline(color="black")
    plt.plot(data_x, data_y)
    hold.pyplot(fig, use_container_width=True)
#######################################################################################################################

                                                    #MAIN

#######################################################################################################################
st.markdown("## Futures' Movements (FUTSTK)")
if "top_menu" not in st.session_state:
    st.session_state.top_menu = 'All'

all_stocks_data = extract_bhavcopy(option)
nift = extract_bhavcopy('FUTIDX')
#nift = nift.loc[(nift['SYMBOL'] == 'NIFTY') | (nift['SYMBOL'] == 'BANKNIFTY') | (nift['SYMBOL'] == 'FINNIFTY')].reset_index(drop=True)
all_stocks_data = pd.concat([all_stocks_data, nift], axis=0).reset_index(drop=True)
action_data_all_stk = pd.DataFrame()
for i in stk_symbol_list:
    one_symbol_data = all_stocks_data.loc[(all_stocks_data['SYMBOL'] == i)].reset_index(drop=True)
    eexp = list(set(list(one_symbol_data['EXPIRY_DT'])))[0]
    print("-------------------",eexp)
    #one_symbol_data = one_symbol_data.loc[one_symbol_data['EXPIRY_DT'] == ]
    action_data_one_stk = action_setting(one_symbol_data)
    action_data_all_stk = pd.concat([action_data_all_stk, action_data_one_stk], ignore_index=True)

# st.dataframe(action_data_all_stk)
frag(1, action_data_all_stk)
