import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import datetime, timedelta
from app_funcs import get_forecast, predict_biking

st.set_page_config(page_title="Predicting NYC Bike Ridership", layout="wide")

# create default states
tomorrow = datetime.today() + timedelta(days=1)
states = ['date', 'prcp', 'tmax', 'tmin', 'rad', 'day_precip', 'day_real_feel',
                            'day_wind', 'prev_count']
for state in states:
    if state not in st.session_state:
        if state=='date':
            st.session_state[state] = tomorrow
        else: 
            st.session_state[state] = str(0.)

st.title("Predicting NYC Bike Ridership")
st.write('This project uses (predominantly) weather data to predict daily bike ridership across New York City. New York City\
         publishes bike ridership data via it\'s open data portal. I used the total daily counts across six primary bike\
          counters and incorporated hourly weather data to train a predictive model to forecast the following day\'s ridership.')
st.write('')
st.write('The model utilizes gradient boosted trees via the xgboost library.')
st.write('')
st.write('The below charts show prediction results for two months in 2023. These months were witheld from model training and validation so \
         the model has never seen them before.')
test_df = pd.read_pickle('data/test_df.pkl')
jan = test_df[test_df['date'].dt.month==1].copy()
jan = pd.melt(jan[['date', 'actual', 'predicted']], id_vars=['date'], value_vars=['actual', 'predicted'])
aug = test_df[test_df['date'].dt.month==8].copy()
aug = pd.melt(aug[['date', 'actual', 'predicted']], id_vars=['date'], value_vars=['actual', 'predicted'])

### CHARTS
hover_selection = alt.selection_point(on='mouseover', fields=['date'], nearest=True)

# JAN
jan_chart = alt.Chart(jan).mark_line().encode(
    x=alt.X('date:T', axis=alt.Axis(title='day of month', format='%d', grid=True)),
    y=alt.Y('value:Q', title='riders per day'),
    color=alt.Color('variable:N', scale=alt.Scale(scheme='set2')),
    opacity=alt.condition(hover_selection, alt.value(1), alt.value(0.4))
    ).add_params(hover_selection)

# Create a selection that chooses the nearest point & selects based on x-value
nearest_date = alt.selection_point(nearest=True, on='mouseover',
                        fields=['date'], empty=False)

# transparent selectors across the chart. This is what tells us
# the x-value of the cursor
selectors = alt.Chart(jan).mark_point().encode(
    x='date:T',
    opacity=alt.value(0),
    tooltip=alt.value(None)
).add_params(
    nearest_date
)

# traw points on the line, and highlight based on selection
points = jan_chart.mark_point().encode(
    opacity=alt.condition(nearest_date, alt.value(1), alt.value(0))
)

# draw text labels near the points, and highlight based on selection
text = jan_chart.mark_text(align='left', dx=10, dy=10).encode(
    text=alt.condition(nearest_date, alt.Text('value:Q', format='.0f'), alt.value(' '))
)

# draw a rule at the location of the selection
rules = alt.Chart(jan).mark_rule(color='gray').encode(
    x='date:T'
).transform_filter(
    nearest_date
)

# Put the five layers into a chart and bind the data
jan_bound = alt.layer(
    jan_chart, selectors, points, rules, text
)

# aug
aug_chart = alt.Chart(aug).mark_line().encode(
    x=alt.X('date:T', axis=alt.Axis(title='day of month', format='%d', grid=True)),
    y=alt.Y('value:Q', title=None),
    color=alt.Color('variable:N', scale=alt.Scale(scheme='set3')),
    opacity=alt.condition(hover_selection, alt.value(1), alt.value(0.4)),
    ).add_params(hover_selection)


# Create a selection that chooses the nearest point & selects based on x-value
nearest_date = alt.selection_point(nearest=True, on='mouseover',
                        fields=['date'], empty=False)

# Transparent selectors across the chart. This is what tells us
# the x-value of the cursor
selectors = alt.Chart(aug).mark_point().encode(
    x='date:T',
    opacity=alt.value(0),
    tooltip=alt.value(None)
).add_params(
    nearest_date
)

# Draw points on the line, and highlight based on selection
points = aug_chart.mark_point().encode(
    opacity=alt.condition(nearest_date, alt.value(1), alt.value(0))
)

# Draw text labels near the points, and highlight based on selection
text = aug_chart.mark_text(align='left', dx=10, dy=10).encode(
    text=alt.condition(nearest_date, alt.Text('value:Q', format='.0f'), alt.value(' '))
)

# Draw a rule at the location of the selection
rules = alt.Chart(aug).mark_rule(color='gray').encode(
    x='date:T'
).transform_filter(
    nearest_date
)

aug_bound = alt.layer(
    aug_chart, selectors, points, rules, text
)

col1, col2 = st.columns(2)
col1.markdown("<h4 style='text-align: center;'>Jan 2023 </h4>", unsafe_allow_html=True)
col1.altair_chart(jan_bound, use_container_width=True)
col2.markdown("<h4 style='text-align: center;'>Aug 2023</h4>", unsafe_allow_html=True)
col2.altair_chart(aug_bound, use_container_width=True)

### FORECAST

st.markdown("<p></p>", unsafe_allow_html=True)
st.markdown("<p></p>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left;'>Make your own prediction! </h4>", unsafe_allow_html=True)
st.markdown("<p></p>", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown("<p></p>", unsafe_allow_html=True)
    if st.button("fill with tomorrow's forecast"):
        wthr = get_forecast()
        for col in wthr.columns:
            if col =='date':
                st.session_state[col] = datetime.utcfromtimestamp(wthr[col].values[0].astype('int64') * 1e-9)
            else:
                st.session_state[col] = str(wthr[col].values[0])
    st.write('or fill in your own data:')

with c2:
    st.markdown("<p></p>", unsafe_allow_html=True)
    date = st.date_input(label='select date',
                        key='date')
    prev_count = st.text_input("previous day ridership",
                               placeholder='try 17297, the daily average',
                               key='prev_count')
    tmax = st.text_input("high temp (F)", 
                         placeholder='high temp',
                         key='tmax')
    tmin = st.text_input("low temp (F)", 
                         placeholder='low temp',
                         key='tmin')
    day_rf = st.text_input("daytime realfeel (F)", 
                           placeholder='avg. daytime "feels like"',
                           key='day_real_feel')

with c3:
    st.markdown("<p></p>", unsafe_allow_html=True)
    precip = st.text_input("total precipitation (in)", 
                           placeholder='total 24 hr precip.',
                           key='prcp')
    day_precip = st.text_input("daytime precipitation (in)", 
                               placeholder='total daytime precip.',
                               key='day_precip')
    day_wind = st.text_input("avg. wind speed (mph)", 
                             placeholder='avg. daytime wind speed',
                             key='day_wind')
    rad = st.text_input("total solar radiation (W/m2)", 
                        placeholder='how sunny it is, try 2700',
                        key='rad')

with c5:
    st.markdown("<p></p>", unsafe_allow_html=True)
    predict = st.button("predict", on_click=predict_biking())
    if predict:
        pred = st.session_state['pred']
        st.markdown(f"<h1 style='text-align: left;'>{pred:0.0f} bikers", unsafe_allow_html=True)
        if np.round(pred, 0) % 2 == 0:
            st.markdown(f"<h1 style='text-align: left;'>&#127881 &#128692;&#8205;&#9792;&#65039;</h1>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='text-align: left;'>&#127881 &#128692</h1>", unsafe_allow_html=True)

        

st.markdown("<h4 style='text-align: left;'>Background & sources</h4>", unsafe_allow_html=True)
li = 'https://www.linkedin.com/in/nliusont/'
st.write('This streamlit app and underlying model were developed \
         by [Nick Liu-Sontag](%s), a data scientist :nerd_face: in Brooklyn, NY' % li)

od = 'https://data.cityofnewyork.us/Transportation/Bicycle-Counts/uczf-rk3c'
om = 'https://open-meteo.com/'
noa = 'https://www.ncdc.noaa.gov/cdo-web/webservices/v2#dataTypes'
st.write('Sources: ')
st.write('[NYC Open Data](%s)' % od)
st.write('[Open Mateo Weather API](%s)' % om)
st.write('[NOAA Climate Data Online API](%s)' % noa)


