## pull forecast
def get_forecast():
    '''gets weather forecast and returns df with which to predict'''

    import requests
    import json
    import pickle
    from datetime import datetime, timedelta
    import pandas as pd
    from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
    import streamlit as st
    import numpy as np

    today = datetime.today()
    start_date = today + timedelta(days=1)
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = today + timedelta(days=2)
    end_date = end_date.strftime("%Y-%m-%d")

    url = f'https://api.open-meteo.com/v1/forecast?latitude=40.7761&longitude=-73.8727&hourly=temperature_2m,apparent_temperature,precipitation,windspeed_10m,direct_radiation,is_day&temperature_unit=fahrenheit&windspeed_unit=mph&precipitation_unit=inch&timezone=America%2FNew_York&start_date={start_date}&end_date={end_date}'
    r = requests.get(url)
    d = json.loads(r.text)

    wthr = pd.DataFrame({
        'date':d['hourly']['time'],
        'is_day':d['hourly']['is_day'],
        'rad':d['hourly']['direct_radiation'],
        'prcp':d['hourly']['precipitation'],
        'temp':d['hourly']['temperature_2m'],
        'real_feel':d['hourly']['apparent_temperature'],
        'wind':d['hourly']['windspeed_10m']
        }).dropna()

    wthr['tmin'] = wthr['temp']
    wthr['tmax'] = wthr['temp']
    wthr['day_precip'] = wthr['is_day'] * wthr['prcp']
    wthr['day_real_feel'] = wthr['is_day'] * wthr['real_feel']
    wthr['day_wind'] = wthr['is_day'] * wthr['wind']
    wthr.loc[wthr['is_day']==0., 'day_real_feel'] = np.nan
    wthr.loc[wthr['is_day']==0., 'day_wind'] = np.nan
    wthr['date'] = pd.to_datetime(wthr['date'])
    wthr = wthr[['date','prcp', 'tmax', 'tmin', 'rad', 'day_precip', 'day_real_feel',
        'day_wind']]
    
    wthr = wthr.groupby(wthr['date'].dt.date).agg({
    'tmax':'max',
    'tmin':'min',
    'rad':'sum',
    'prcp':'sum',
    'day_precip':'sum',
    'day_real_feel':'mean',
    'day_wind':'mean'
    }).reset_index()
    wthr['prev_count'] = 17297 # use average of dataset

    # day of week
    wthr['date'] = pd.to_datetime(wthr['date'])
    wthr['year'] = wthr['date'].dt.year
    wthr['month'] = wthr['date'].dt.month
    wthr['date'] = pd.to_datetime(wthr['date'])
    wthr['dow'] = wthr['date'].dt.dayofweek
    wthr['dom'] = wthr['date'].dt.day

    # holidays
    cal = calendar()
    holidays = cal.holidays(start=wthr['date'].min(), end=wthr['date'].max())
    wthr['hol'] = wthr['date'].isin(holidays)

    for col in wthr.columns:
        if (col=='date') or (col=='hol'):
            continue
        else:
            wthr[col] = pd.to_numeric(wthr[col])
            wthr[col] = wthr[col].round(1)

    wthr = wthr[['date', 'prcp', 'tmax', 'tmin', 'rad', 'day_precip', 'day_real_feel', 'day_wind', 'year', 'month', 'dow', 'dom', 'hol', 'prev_count']].copy()
    return wthr

def predict_biking():
    '''takes weather forecast and predicts ridership'''
    import pickle
    import pandas as pd
    import streamlit as st
    from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

    #import model
    with open('model/xgb_v1.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.DataFrame(index=[0])

    for k, v in st.session_state.items():
        if k=='pred':
            continue
        elif k=='date':
            df[k] = v
        elif k=='hol':
            df[k] = bool(v)
        else:
            df[k] = float(v)
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dow'] = df['date'].dt.dayofweek
    df['dom'] = df['date'].dt.day
    # holidays
    cal = calendar()
    holidays = cal.holidays(start=df['date'].min(), end=df['date'].max())
    df['hol'] = df['date'].isin(holidays)

    df = df[['prcp', 'tmax', 'tmin', 'rad', 'day_precip', 'day_real_feel', 'day_wind', 'year', 'month', 'dow', 'dom', 'hol', 'prev_count']]
    pred = model.predict(df)[0]
    st.session_state['pred'] = pred