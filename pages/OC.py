import requests
import pandas as pd
import numpy as np
import time
import xlwings as xw
from bs4 import BeautifulSoup
import datetime
import streamlit as st
import csv

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.sidebar.header("OC")
exchange = "NSE"
ATM = 0


def last_thursdays(year):
    exp = []
    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        if month == 1 or month == 2 or month == 3 or month == 4 or month == 5 or month == 6 or month == 7 or month == 8 or month == 9:
            date = f"{year}-0{month}-01"
        if month == 10 or month == 11 or month == 12:
            date = f"{year}-{month}-01"

        # we have a datetime series in our dataframe...
        df_Month = pd.to_datetime(date)

        # we can easily get the month's end date:
        df_mEnd = df_Month + pd.tseries.offsets.MonthEnd(1)

        # Thursday is weekday 3, so the offset for given weekday is
        offset = (df_mEnd.weekday() - 3) % 7

        # now to get the date of the last Thursday of the month, subtract it from
        # month end date:
        df_Expiry = df_mEnd - pd.to_timedelta(offset, unit='D')
        exp.append(df_Expiry.date())

    return exp


today_year = datetime.datetime.now().year
exp_date_list = last_thursdays(today_year)
DATE_LIST = []
TODAY = datetime.date.today()
for i in range(len(exp_date_list)):
    x = (exp_date_list[i] - TODAY).days
    if x >= 0:
        DATE_LIST.append(exp_date_list[i].strftime('%d-%m-%Y'))
EXP_OPTION = DATE_LIST[0]


def current_market_price(ticker, exchange):
    url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"

    for _ in range(1000000):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        class1 = "YMlKec fxKbKc"

        price = float(soup.find(class_=class1).text.strip()[1:].replace(",", ""))
        return price


def fifty_two_week_high_low(ticker, exchange):
    url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    class1 = "P6K39c"

    price = soup.find_all(class_=class1)[2].text
    low_52_week = float(price.split("-")[0].strip()[1:].replace(",", ""))
    high_52_week = float(price.split("-")[1].strip()[1:].replace(",", ""))
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

    atm_price = current_market_price(ticker, exchange)
    diffdata = pd.DataFrame({"Strike": strike, "Difference": [abs(x - atm_price) for x in strike]})
    atm_price = diffdata.loc[diffdata['Difference'] == diffdata['Difference'].min()]['Strike'].head(1).item()

    five_percent_cmp_ce = atm_price - 0.05 * atm_price
    five_percent_cmp_pe = atm_price + 0.05 * atm_price

    # access dataframe for atm price
    diffdata_ce = pd.DataFrame({"Strike": strike, "Difference": [abs(x - five_percent_cmp_ce) for x in strike]})
    atm_ce = diffdata_ce.loc[diffdata_ce['Difference'] == diffdata_ce['Difference'].min()]['Strike'].head(1).item()

    diffdata_pe = pd.DataFrame({"Strike": strike, "Difference": [abs(x - five_percent_cmp_pe) for x in strike]})
    atm_pe = diffdata_pe.loc[diffdata_pe['Difference'] == diffdata_pe['Difference'].min()]['Strike'].head(1).item()

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
    ind_atm_ce = subset_ce[(subset_ce.strikePrice == atm_ce)].index.tolist()[0]
    ind_atm_pe = subset_ce[(subset_ce.strikePrice == atm_pe)].index.tolist()[0]
    subset_ce = subset_ce.loc[ind_atm_ce:ind_atm_pe,].reset_index(drop=True)
    output_ce = pd.concat([output_ce, subset_ce]).reset_index(drop=True)
    output_ce = categorize(output_ce)

    # (subset_pe (PE))
    subset_pe = fd_pe[(fd_pe.instrumentType == "PE") & (fd_pe.expiryDate == adjusted_expiry_pe)].reset_index(drop=True)
    ind_atm_ce = subset_pe[(subset_pe.strikePrice == atm_ce)].index.tolist()[0]
    ind_atm_pe = subset_pe[(subset_pe.strikePrice == atm_pe)].index.tolist()[0]
    subset_pe = subset_pe.loc[ind_atm_ce:ind_atm_pe, ].reset_index(drop=True)
    print("CPE: ",atm_ce, atm_pe)
    output_pe = pd.concat([output_pe, subset_pe]).reset_index(drop=True)
    output_pe = categorize(output_pe)

    output_ce.reset_index(drop=True, inplace=True)
    output_pe.reset_index(drop=True, inplace=True)

    return output_ce, output_pe, atm_price


def highlight_background(s):
    global ATM
    atm = ATM
    if s.STR_PRICE >= atm:
        col = ['background-color: white']*7+['background-color: antiquewhite']*6
    elif s.STR_PRICE < atm:
        col = ['background-color: antiquewhite']*6+['background-color: white']*7
    return col


def highlight_text(s):
    cat_ce = s.Category_CE
    cat_pe = s.Category_PE
    colce, colpe = '', ''
    if cat_ce == 'LB' or cat_ce == 'SC':
        colce = 'green'
    else:
        colce = 'red'
    if cat_pe == 'LB' or cat_pe == 'SC':
        colpe = 'green'
    else:
        colpe = 'red'
    if colce == 'green' and colpe == 'green':
        col = ['color: black']*5+['color: green']+['color: black']*6+['color: green']
    elif colce == 'green' and colpe == 'red':
        col = ['color: black']*5+['color: green']+['color: black']*6+['color: red']
    elif colce == 'red' and colpe == 'green':
        col = ['color: black']*5+['color: red']+['color: black']*6+['color: green']
    else:
        col = ['color: black']*5+['color: red']+['color: black']*6+['color: red']
    return col

@st.experimental_fragment
def frag_table(table_number, selected_option='UBL', exp_option=EXP_OPTION):
    global ATM
    global OPT
    shares = pd.read_csv("FNO Stocks - All FO Stocks List, Technical Analysis Scanner.csv")
    share_list = list(shares["Symbol"])
    share_list.sort()
    selected_option = selected_option.strip()
    share_list.remove(selected_option)
    share_list = [selected_option] + share_list

    exp_date_list_sel = DATE_LIST.copy()
    print("LIST: ", exp_date_list_sel)
    exp_option = datetime.datetime.strptime(exp_option, "%d-%m-%Y").date().strftime('%d-%m-%Y')
    print("EXP_OPTION:", exp_option)
    exp_date_list_sel.remove(exp_option)
    exp_date_list_sel = [exp_option] + exp_date_list_sel
    #
    # date_list = []
    # today_date = datetime.date.today()
    # for i in range(len(exp_date_list)):
    #     x = (exp_date_list[i] - today_date).days
    #     if x > 0:
    #         date_list.append(exp_date_list[i].strftime('%d-%m-%Y'))
    c1, c2 = st.columns(2)
    with c1:
        selected_option = st.selectbox("Share List", share_list, key="share_list" + str(table_number))
    with c2:
        exp_option = st.selectbox("Expiry Date", exp_date_list_sel, key="exp_list" + str(table_number))
        if selected_option in share_list:
            ticker = selected_option
            output_ce, output_pe, atm = get_dataframe(ticker, exp_option)
            ATM = atm
            ########################################## Stock LTP and Matrix #######################################
            stock_ltp = current_market_price(ticker, exchange)
            low_52_week, high_52_week = fifty_two_week_high_low(ticker, exchange)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.write(f'CMP:', stock_ltp)
    with d2:
        st.write(f'52 week low:', low_52_week)
    with d3:
        st.write(f'52 week high:', high_52_week)

    output_ce = output_ce[['strikePrice', 'pchangeinOpenInterest', 'pChange', 'totalTradedVolume', 'impliedVolatility', 'lastPrice', 'Category']]
    output_ce = output_ce.rename(columns={'strikePrice': 'STR_PRICE',
                                          'pchangeinOpenInterest': 'OI%_CE',
                                          'totalTradedVolume': 'Volume_CE',
                                          'impliedVolatility': 'IV_CE',
                                          'lastPrice': 'LTP_CE',
                                          'pChange': '%Change_CE',
                                          'Category': 'Category_CE'})
    output_pe = output_pe[['strikePrice', 'pchangeinOpenInterest', 'pChange', 'totalTradedVolume', 'impliedVolatility', 'lastPrice',
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
    output = output[cols[1:7]+[cols[0]]+cols[7:]]
    output = output.style.apply(highlight_background, axis=1)
    output = output.apply(highlight_text, axis=1)
    output = output.set_properties(
        **{'text-align': 'center'}).set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'center')]}])
    output = output.format({'OI%_CE': "{:.2f}".format, '%Change_CE': "{:.1f}".format, 'LTP_CE': "{:.2f}".format,
                            'IV_CE': "{:.2f}".format, 'OI%_PE': "{:.2f}".format, '%Change_PE': "{:.1f}".format,
                            'LTP_PE': "{:.2f}".format, 'IV_PE': "{:.2f}".format})
    st.markdown('<style>.col_heading{text-align: center}</style>', unsafe_allow_html=True)
    output.columns = ['<div class="col_heading">' + col + '</div>' for col in output.columns]

    st.write(output.to_html(escape=False), unsafe_allow_html=True, use_container_width=True)

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


hist = pd.read_csv("history.csv")
hist_df = pd.DataFrame(hist)

print(len(hist_df))

if len(hist_df) > 0:
    last_rec = hist_df.tail(1)
    print(last_rec)
    frag_table(1, last_rec['table1'].item(), last_rec['exp1'].item())
    #frag_table(2, last_rec['table2'].item(), last_rec['exp2'].item())
    #frag_table(3, last_rec['table3'].item(), last_rec['exp3'].item())
else:
    frag_table(1, 'RELIANCE')
    #frag_table(2, 'VEDL')
    #frag_table(3, 'INFY')
