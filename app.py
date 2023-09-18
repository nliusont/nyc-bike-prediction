import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import datetime, timedelta
from app_funcs import get_forecast, predict_biking
import pickle

st.set_page_config(page_title="Predicting NYC Bike Ridership", layout="wide")

# create default states
tomorrow = datetime.today() + timedelta(days=1)
states = ['date', 'prcp', 'tmax', 'tmin', 'rad', 'day_precip', 'day_real_feel',
                            'day_wind', 'prev_count', 'last_count']
for state in states:
    if state not in st.session_state:
        if state=='date':
            st.session_state[state] = tomorrow
        else: 
            st.session_state[state] = str(0.)

with open('data/feature_dict.pkl', 'rb') as f:
    input_features = pickle.load(f)

st.title("Predicting NYC Bike Ridership")
st.write('This is a model that predicts daily bike ridership across six of the primary bike\
          counters in New York City. It turns out most day-to-day ridership is determined by the weather.')
st.write('The model utilizes the xgboost python library. The primary source of inputs is hourly weather data.\
         Scroll down for more of the geeky details on methodology.')
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
        st.write('')
        last_count = float(st.session_state['last_count'])
        delta = (pred - last_count) / last_count * 100
        if delta==np.nan:
            delta=0.0
        st.metric('bikers:', f'{pred:0.0f}', f'{np.round(delta, 1)}%')
        if np.round(pred, 0) % 2 == 0:
            st.markdown(f"<h2 style='text-align: left;'>&#127881 &#128692;&#8205;&#9792;&#65039;</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='text-align: left;'>&#127881 &#128692</h2>", unsafe_allow_html=True)

column1, column2 = st.columns(2)

with column1:
    st.markdown("<h4 style='text-align: left;'>Methodology</h4>", unsafe_allow_html=True)
    gh = 'https://github.com/nliusont/nyc-bike-counts'
    st.write('You can find the notebooks that I used to prepare this data and model \
        [here](%s).' % gh)
    st.markdown("<h5 style='text-align: left;'>Data</h5>", unsafe_allow_html=True)    
    st.write('Bike count data for this project came from NYC\'s Open Data Portal (link below) where the NYCDOT publishes \
            15-min interval data from bike counters placed across the city. The data for some counters goes back as \
            far as ten years.')
    st.write('For this project, I focused on the predominant bike counters that are placed on critical bike paths and bridges. \
            The below counters are the ones that were used for this analysis.')
    st.markdown("<div><i>Bike counters</i></div>", unsafe_allow_html=True)  
    with open('data/counter_dict.pkl', 'rb') as f:
        counters = pickle.load(f)
    st.write(counters)
    st.markdown("<div style='font-size: 12px;'><i>Kent Ave is a single counter that moved and is marked by two ids: 100058279 & 100010019</i></div>", unsafe_allow_html=True)  
    st.write('')
    st.write('The bike count data was retrieved on 9/13/23. I worked with daily data by grouping by the date and summing the total counts for each day. \
            The dataset was limited to the dates between 1/1/14 and 9/1/23 as this is when all six counters were active. \
            This yielded 3,538 days of data. While this is a very small dataset for any machine learning model, it does exhibit strong patterns \
            which appear to make up for its small size.')
    st.markdown("<h5 style='text-align: left;'>Features</h5>", unsafe_allow_html=True)  
    st.write('The input weather features were retrieved via historical weather APIs from Open Mateo and NOAA. The final model uses the below features.')
    st.markdown("<div><i>Input features</i></div>", unsafe_allow_html=True)  
    st.write(input_features)
    st.write('Historical weather from Open Mateo was grouped by day and either summed or averaged depending on the feature. \
            In initial testing, it became clear that the model could not account for extreme weather that ocurrs in the nighttime. \
            This makes sense since most biking occurs during the day, and a nighttime thunderstorm has little impact on daytime ridership. \
            Thankfully, Open Mateo provides the feature "is_day", which records whether a given hour is daytime. I used this \
            feature as a mask to produce the daytime features listed above.')
    st.markdown("<div><i>Training</i></div>", unsafe_allow_html=True) 
    st.write('At the outset of training, Jan 2023 and Aug 2023 were witheld as test sets. The remaining data was split \
            into training (70%) and validation (30%) sets. Initial testing was performed with a range of regressors running with \
            default hyperparameters. Xgboost was the clear favorite.')
    st.write('I then went through uncountable rounds of feature selection and hyperparameter tuning (there were many more features than what is listed above).')
    st.markdown("<div><i>Performance</i></div>", unsafe_allow_html=True) 
    st.write('The below tables list the model\'s performance on each set of data')
    with open('data/performance.pkl', 'rb') as f:
        perf = pickle.load(f)
    st.write(perf)
    st.write('The MAE scores indicate a fair amount of overfitting to the training set, however, the results are still reasonably accurate.')

    st.markdown("<h4 style='text-align: left;'>Background & sources</h4>", unsafe_allow_html=True)
    li = 'https://www.nls.website/'
    st.write('This streamlit app and underlying model were developed \
            by [Nick Liu-Sontag](%s), a data scientist :nerd_face: in Brooklyn, NY' % li)

    od = 'https://data.cityofnewyork.us/Transportation/Bicycle-Counts/uczf-rk3c'
    om = 'https://open-meteo.com/'
    noa = 'https://www.ncdc.noaa.gov/cdo-web/webservices/v2#dataTypes'
    st.write('Sources: ')
    st.write('[NYC Open Data](%s)' % od)
    st.write('[Open Mateo Weather API](%s)' % om)
    st.write('[NOAA Climate Data Online API](%s)' % noa)


