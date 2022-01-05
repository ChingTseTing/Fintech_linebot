# -*- coding: utf-8 -*-
#載入LineBot所需要的套件
from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *

import numpy as np 
import sys, os
import pyimgur
import yfinance as yf
import pandas as pd
import psycopg2
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import talib as ta
from talib import abstract
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout,BatchNormalization
import keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import requests
from bs4 import BeautifulSoup
# import configparser
# from dotenv import load_dotenv
# load_dotenv()

app = Flask(__name__)


# LINE 聊天機器人的基本資料
# config = configparser.ConfigParser()
# config.read('config.ini')
# # 必須放上自己的Channel Access Token
# line_bot_api = LineBotApi(config.get('line-bot', 'channel_access_token'))
# # 必須放上自己的Channel Secret
# handler = WebhookHandler(config.get('line-bot', 'channel_secret'))

# 必須放上自己的Channel Access Token
line_bot_api = LineBotApi(os.environ["CHANNEL_ACCESS_TOKEN"])
# 必須放上自己的Channel Secret
handler = WebhookHandler(os.environ["CHANNEL_SECRET"])



# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

# 功能模組-求股價
def get_stock(stock_id ,PERIOD , INTERVAL ):
  tmp = yf.download(stock_id , period=PERIOD ,interval=INTERVAL )# start='2016-01-01',end=datetime.now().strftime('%Y-%m-%d')
  return tmp

# 功能模組-求index
def get_stock_index(df):
  # Bias of moving average
  df['bias_MA_5'] =  ( df['Close']- ta.SMA(df['Close'], timeperiod=5)  ) / ta.SMA(df['Close'], timeperiod=5)
  df['bias_MA_10'] = ( df['Close']- ta.SMA(df['Close'], timeperiod=10)  ) / ta.SMA(df['Close'], timeperiod=10)
  df['bias_MA_20'] = ( df['Close']- ta.SMA(df['Close'], timeperiod=20)  ) / ta.SMA(df['Close'], timeperiod=20)
  df['bias_MA_60'] = ( df['Close']- ta.SMA(df['Close'], timeperiod=60)  ) / ta.SMA(df['Close'], timeperiod=60)
  # RSI_14
  df['RSI_14']    = ta.RSI(df['Close'], timeperiod=14)
  # DIF DEM HIST (MACD)
  df['MACD_DIF'], df['MACD_SIGNAL'], df['MACD_BAR'] = ta.MACD( df['Close']  , fastperiod=12, slowperiod=26, signalperiod=9)
  # KD value 
  df['k'], df['d'] =ta.STOCH(df['High'], df['Low'], df['Close'])
  # remove nan
  df.dropna(axis=0,how='any', inplace=True)
  return df # Open	High	Low	Close Adj Close	Volume	MA_5	MA_10	MA_20	MA_60	RSI_14	DIF	MACD	MACD_BAR


def analysis_plot(record):
  tt =  get_stock( record[1] ,record[2] , record[3] )
  # Create subplots and mention plot grid size
  fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=( record[1] , ''), row_width=[0.3, 0.3,1])
  # Plot OHLC on 1st row
  fig.add_trace(go.Candlestick(x=tt.index, open=tt["Open"], high=tt["High"],low=tt["Low"], close=tt["Close"], name="OHLC", showlegend=True ), row=1, col=1 )
  fig.add_trace(go.Scatter(x=tt.index, y=ta.SMA( np.array(tt['Close']) ,timeperiod=5), mode='lines' ,name='MA5', showlegend=True)  , row=1, col=1 )
  fig.add_trace(go.Scatter(x=tt.index, y=ta.SMA( np.array(tt['Close']) ,timeperiod=10), mode='lines' ,name='MA10', showlegend=True)  , row=1, col=1 )
  fig.add_trace(go.Scatter(x=tt.index, y=ta.SMA( np.array(tt['Close']) ,timeperiod=20), mode='lines' ,name='MA20', showlegend=True)  , row=1, col=1 )
  fig.add_trace(go.Scatter(x=tt.index, y=ta.SMA( np.array(tt['Close']) ,timeperiod=60), mode='lines' ,name='MA60', showlegend=True)  , row=1, col=1 )
  # Bar trace for volumes on 2nd row without legend
  fig.add_trace(go.Bar(x=tt.index, y=tt['Volume'], showlegend=False), row=2, col=1)
  # plot index
  if record[4]=="RSI":
    fig.add_trace(go.Scatter(x=tt.index, y=ta.RSI(tt['Close']), mode='lines' ,name=record[4], showlegend=True)  , row=3, col=1 ) #ta.RSI( np.array(tt['Close']) ,timeperiod=14)
    fig['layout']['yaxis3']['title']='RSI' 
  if record[4]=="KD":
    K,D = ta.STOCH(tt['High'], tt['Low'], tt['Close'])
    fig.add_trace(go.Scatter(x=tt.index, y=K, mode='lines' ,name='K', showlegend=True)  , row=3, col=1 ) #ta.RSI( np.array(tt['Close']) ,timeperiod=14)
    fig.add_trace(go.Scatter(x=tt.index, y=D, mode='lines' ,name='D', showlegend=True)  , row=3, col=1 ) #ta.RSI( np.array(tt['Close']) ,timeperiod=14)
    fig['layout']['yaxis3']['title']='KD'
  if record[4]=="MACD":
    MACD_DIF , MACD_SIGNAL, MACD_BAR = ta.MACD(tt['Close'])
    fig.add_trace(go.Scatter(x=tt.index, y=MACD_DIF, mode='lines' ,name='MACD_DIF', showlegend=True)  , row=3, col=1 ) #ta.RSI( np.array(tt['Close']) ,timeperiod=14)
    fig.add_trace(go.Scatter(x=tt.index, y=MACD_SIGNAL, mode='lines' ,name='MACD_SIGNAL', showlegend=True)  , row=3, col=1 ) #ta.RSI( np.array(tt['Close']) ,timeperiod=14)
    fig.add_trace(go.Bar(x=tt.index, y=MACD_BAR,name='MACD_BAR', showlegend=True)  , row=3, col=1 ) #ta.RSI( np.array(tt['Close']) ,timeperiod=14)
    fig['layout']['yaxis3']['title']='MACD'

  fig['layout']['yaxis']['title']='Price'
  fig['layout']['yaxis2']['title']='Volume'

  fig.update_layout(margin=dict(l=30, r=30, t=30, b=30) , template='plotly_dark',paper_bgcolor ='rgb(10,10,10)')
  # Do not show OHLC's rangeslider plot 
  fig.update(layout_xaxis_rangeslider_visible=False)
  # save image
  fig.write_image("send.png")
  CLIENT_ID = "08680019f3643c6"  #"TingChingTse"
  PATH = "send.png"
  im = pyimgur.Imgur(CLIENT_ID)
  uploaded_image = im.upload_image(PATH, title="Uploaded with PyImgur")
  return uploaded_image.link


def LSTM_model(record):
  new_df = get_stock_index( get_stock(record[1] ,'9mo' , '1d' )   )
  #切分Test集
  train_percent = 0.7
  train = new_df.head(int(new_df.shape[0]*train_percent))
  test_percent = 1-train_percent
  test = new_df.tail(int(new_df.shape[0]*test_percent))
  train_set = train['Close']
  test_set = test['Close']
  sc = MinMaxScaler(feature_range = (0, 1))
  #需將資料做reshape的動作，使其shape為(資料長度,1) 
  train_set= train_set.values.reshape(-1,1)
  training_set_scaled = sc.fit_transform(train_set)  #train_set(stock price) normalize
  X_train = [] 
  y_train = []
  for i in range(10,len(train_set)):
    X_train.append(training_set_scaled[i-10:i-1, 0])  #第i天前的stock price
    y_train.append(training_set_scaled[i, 0])  #第i天的stock price
  X_train, y_train = np.array(X_train), np.array(y_train) 
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
  keras.backend.clear_session()
  # Initialising the RNN
  regressor = Sequential()
  # Adding the first LSTM layer
  regressor.add(LSTM(units = 100, return_sequences=True, input_shape = (X_train.shape[1], 1))) #units=number of neurons
  regressor.add(Dropout(0.2))
  # Adding a second LSTM layer
  regressor.add(LSTM(units = 100))
  regressor.add(Dropout(0.2))
  regressor.add(Dense(units = 1))  #Ouput Layer
  regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') #optimizer:Adam, loss:MSE


  stringlist = []
  regressor.summary(print_fn=lambda x: stringlist.append(x))
  short_model_summary = "\n".join(stringlist)


  # type(regressor.summary())
  # start training
  history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 16)
  dataset_total = pd.concat((train['Close'], test['Close']), axis = 0)
  inputs = dataset_total[len(dataset_total) - len(test) - 10:].values
  inputs = inputs.reshape(-1,1)
  inputs = sc.transform(inputs)
  X_test = []
  for i in range(10, len(inputs)):
    X_test.append(inputs[i-10:i-1, 0])
  X_test = np.array(X_test)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  predicted_stock_price = regressor.predict(X_test)
  #使用sc的 inverse_transform將股價轉為歸一化前

  predicted_stock_price = sc.inverse_transform(predicted_stock_price)

  #預測
  test_test=[]
  test_test.append(inputs[len(inputs)-10:len(inputs)-1, 0])
  test_test= np.reshape(test_test, (1, 9, 1))
  predicted_stock_price1 = regressor.predict(test_test)
  #使用sc的 inverse_transform將股價轉為歸一化前
  predicted_stock_price1 = sc.inverse_transform(predicted_stock_price1)

  #plot
  fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, subplot_titles=( 'train loss' , 'prediction'), row_width=[0.6, 0.4])
  fig.add_trace(go.Scatter(  y=history.history["loss"], mode='lines' ,name='loss', showlegend=True)  , row=1, col=1 ) 
  fig.add_trace(go.Scatter(x=test.index, y=test['Close'].values, mode='lines' ,name='Real Stock Price', showlegend=True)  , row=2, col=1 ) 
  fig.add_trace(go.Scatter(x=test.index, y=predicted_stock_price.reshape(len(predicted_stock_price),), mode='lines' ,name='Predicted Stock Price', showlegend=True)  , row=2, col=1 )
  fig['layout']['yaxis']['title']='loss'
  fig['layout']['xaxis']['title']='epoch'
  fig['layout']['yaxis2']['title']='price'
  fig.update_layout(margin=dict(l=30, r=30, t=30, b=30) , template='plotly_dark',paper_bgcolor ='rgb(10,10,10)')
  fig.update(layout_xaxis_rangeslider_visible=False)

  fig.write_image("send.png")
  CLIENT_ID = "08680019f3643c6"  #"TingChingTse"
  PATH = "send.png"
  im = pyimgur.Imgur(CLIENT_ID)
  uploaded_image = im.upload_image(PATH, title="Uploaded with PyImgur")
  return predicted_stock_price1 ,uploaded_image.link , short_model_summary

def RF_model(record):
  df = get_stock_index( get_stock(record[1] ,'3y' , '1d' )   )
  X_o = df.iloc[:, :].values
  X_o = np.delete(X_o, 3, axis = 1)
  prediction = np.reshape(X_o[len(df)-1],(1,df.shape[1]-1))
  X = np.delete(X_o, len(df)-1, axis = 0)
  Y_o = df['Close'].shift(-1).values
  Y = np.delete(Y_o, len(df)-1, axis = 0)
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
  forest = RandomForestRegressor(n_estimators=1000, criterion="squared_error", random_state=1, n_jobs=-1)
  forest.fit(X_train, y_train)
  y_train_pred = forest.predict(X_train)
  y_test_pred = forest.predict(X_test)
  RMSE = f'RMSE train:{metrics.mean_squared_error(y_train, y_train_pred , squared=False):.3f}  RMSE test:{metrics.mean_squared_error(y_test, y_test_pred , squared=False):.3f}'
  MAPE = f'MAPE train:{metrics.mean_absolute_percentage_error(y_train, y_train_pred):.3f}  MAPE test:{metrics.mean_absolute_percentage_error(y_test, y_test_pred):.3f}'
  MAE = f'MAE train:{metrics.mean_absolute_error(y_train, y_train_pred ):.3f}  MAE test:{metrics.mean_absolute_error(y_test, y_test_pred):.3f}'
  MSE = f'MSE train:{metrics.mean_squared_error(y_train, y_train_pred ):.3f}  MSE test:{metrics.mean_squared_error(y_test, y_test_pred):.3f}'
  R_2 = f'R^2 train:{r2_score(y_train, y_train_pred ):.3f}  R^2 test:{r2_score(y_test, y_test_pred):.3f}'
  all = RMSE + "\n" + MAPE + "\n" + MAE + "\n" + MSE + "\n" +R_2 +"\n"+ "價格預測:" + str(forest.predict(prediction)[0])
  return all



def stock_info(record):
  headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36' }

  flex = {
      "type": "bubble",
      "body": {
        "type": "box",
        "layout": "vertical",
        "contents": [
          {
            "type": "text",
            "text": "台積電(2330) 半導體",
            "weight": "bold",
            "size": "xl",
            "color": "#ffffff"
          },
          {
            "type": "text",
            "text": "606 ▲2(0.33%)",
            "color": "#FF0000",
            "margin": "none",
            "size": "xl",
            "weight": "bold"
          },
          {
            "type": "text",
            "text": "收盤 | 2021/12/27 14:30 更新",
            "color": "#aaaaaa",
            "margin": "xs",
            "size": "sm"
          },
          {
            "type": "box",
            "layout": "baseline",
            "margin": "none",
            "contents": []
          },
          {
            "type": "box",
            "layout": "vertical",
            "margin": "sm",
            "spacing": "xs",
            "contents": [
              {
                "type": "box",
                "layout": "baseline",
                "spacing": "sm",
                "contents": [
                  {
                    "type": "text",
                    "text": "成交量",
                    "color": "#aaaaaa",
                    "size": "sm"
                  },
                  {
                    "type": "text",
                    "color": "#aaaaaa",
                    "size": "sm",
                    "wrap": True,
                    "text": "16,230"
                  }
                ],
                "margin": "md"
              },
              {
                "type": "box",
                "layout": "baseline",
                "spacing": "sm",
                "contents": [
                  {
                    "type": "text",
                    "text": "成交值(億)",
                    "color": "#aaaaaa",
                    "size": "sm",
                    "margin": "none"
                  },
                  {
                    "type": "text",
                    "text": "98.64",
                    "wrap": True,
                    "color": "#aaaaaa",
                    "size": "sm"
                  }
                ]
              },
              {
                "type": "box",
                "layout": "baseline",
                "contents": [
                  {
                    "type": "text",
                    "text": "內盤",
                    "color": "#aaaaaa",
                    "size": "sm"
                  },
                  {
                    "type": "text",
                    "text": "7,275(45.39%)",
                    "color": "#aaaaaa",
                    "size": "sm"
                  }
                ]
              },
              {
                "type": "box",
                "layout": "baseline",
                "contents": [
                  {
                    "type": "text",
                    "text": "外盤",
                    "color": "#aaaaaa",
                    "size": "sm"
                  },
                  {
                    "type": "text",
                    "text": "7,275(45.39%)",
                    "color": "#aaaaaa",
                    "size": "sm"
                  }
                ]
              },
              {
                "type": "box",
                "layout": "baseline",
                "contents": [
                  {
                    "type": "text",
                    "text": "本益比 (同業平均)",
                    "color": "#aaaaaa",
                    "size": "sm"
                  },
                  {
                    "type": "text",
                    "text": "27.33 (31.61)",
                    "color": "#aaaaaa",
                    "size": "sm"
                  }
                ]
              }
            ]
          },
          {
            "type": "text",
            "text": "新聞",
            "size": "xl",
            "weight": "bold",
            "margin": "xxl",
            "color": "#ffffff"
          },
          {
            "type": "button",
            "action": {
              "type": "uri",
              "label": "action1",
              "uri": "http://linecorp.com/1"
            },
            "margin": "sm",
            "height": "sm",
            "gravity": "top",
            "style": "primary",
            "color": "#808080"
          },
          {
            "type": "button",
            "action": {
              "type": "uri",
              "label": "action2",
              "uri": "http://linecorp.com/2"
            },
            "height": "sm",
            "style": "primary",
            "margin": "sm",
            "color": "#808080"
          },
          {
            "type": "button",
            "action": {
              "type": "uri",
              "label": "action3",
              "uri": "http://linecorp.com/3"
            },
            "height": "sm",
            "margin": "sm",
            "style": "primary",
            "color": "#808080"
          },
          {
            "type": "button",
            "action": {
              "type": "uri",
              "label": "action4",
              "uri": "http://linecorp.com/4"
            },
            "margin": "sm",
            "style": "primary",
            "height": "sm",
            "color": "#808080"
          },
          {
            "type": "button",
            "action": {
              "type": "uri",
              "label": ">> 更多",
              "uri": "http://linecorp.com/"
            },
            "margin": "sm",
            "height": "sm",
            "style": "primary",
            "color": "#808080",
            "offsetTop": "none"
          }
        ],
        "backgroundColor": "#404040"
      }
    }


  url = "https://tw.stock.yahoo.com/quote/" + record[1]
  a= requests.get(url ,headers =headers , allow_redirects= True)
  a.encoding='utf-8'
  soup = BeautifulSoup(a.text , 'html.parser' )

  #名稱
  title = soup.find('div',class_='D(f) Ai(c) Mb(6px)' ).find('h1',class_='C($c-link-text) Fw(b) Fz(24px) Mend(8px)').text
  title = title+"(" + soup.find('div',class_='D(f) Ai(c) Mb(6px)' ).find('span',class_='C($c-icon) Fz(24px) Mend(20px)').text + ")"
  title = title + " " + soup.find('div',class_='D(f) Ai(c) Mb(6px)' ).find('div',class_='Flxg(2)').text
  flex['body']['contents'][0]['text']  =  title # print(title)

  # 價格漲跌
  if soup.find('div',class_='D(f) Fld(c) Ai(fs)' ).find('div',class_='D(f) Ai(fe) Mb(4px)').find('span',class_='Fz(32px) Fw(b) Lh(1) Mend(16px) D(f) Ai(c) C($c-trend-up)')!=None:
    pr =  soup.find('div',class_='D(f) Fld(c) Ai(fs)' ).find('div',class_='D(f) Ai(fe) Mb(4px)').find('span',class_='Fz(32px) Fw(b) Lh(1) Mend(16px) D(f) Ai(c) C($c-trend-up)').text + " ▲"
    pr = pr + soup.find('div',class_='D(f) Fld(c) Ai(fs)' ).find('div',class_='D(f) Ai(fe) Mb(4px)').find('span',class_='Fz(20px) Fw(b) Lh(1.2) Mend(4px) D(f) Ai(c) C($c-trend-up)').text
    pr = pr + soup.find('div',class_='D(f) Fld(c) Ai(fs)' ).find('div',class_='D(f) Ai(fe) Mb(4px)').find('span',class_='Jc(fe) Fz(20px) Lh(1.2) Fw(b) D(f) Ai(c) C($c-trend-up)').text
    flex['body']['contents'][1]['text'] = pr # print(pr)
    flex['body']['contents'][1]['color'] = '#FF0000'

  if soup.find('div',class_='D(f) Fld(c) Ai(fs)' ).find('div',class_='D(f) Ai(fe) Mb(4px)').find('span',class_='Fz(32px) Fw(b) Lh(1) Mend(16px) D(f) Ai(c) C($c-trend-up)')==None:
    pr = soup.find('div',class_='D(f) Fld(c) Ai(fs)' ).find('div',class_='D(f) Ai(fe) Mb(4px)').find('span',class_='Fz(32px) Fw(b) Lh(1) Mend(16px) D(f) Ai(c) C($c-trend-down)').text + " ▼"
    pr = pr + soup.find('div',class_='D(f) Fld(c) Ai(fs)' ).find('div',class_='D(f) Ai(fe) Mb(4px)').find('span',class_='Fz(20px) Fw(b) Lh(1.2) Mend(4px) D(f) Ai(c) C($c-trend-down)').text
    pr = pr + soup.find('div',class_='D(f) Fld(c) Ai(fs)' ).find('div',class_='D(f) Ai(fe) Mb(4px)').find('span',class_='Jc(fe) Fz(20px) Lh(1.2) Fw(b) D(f) Ai(c) C($c-trend-down)').text
    flex['body']['contents'][1]['text'] = pr # print(pr)
    flex['body']['contents'][1]['color'] = '#00FF00'

  # 時間
  flex['body']['contents'][2]['text'] = soup.find('div',class_='D(f) Fld(c) Ai(fs)' ).find('span',class_='C(#6e7780) Fz(12px) Fw(b)').text
  #成交量
  flex['body']['contents'][4]['contents'][0]['contents'][0]['text'] = soup.find('div',class_='D(f) Fld(c) Ai(c) Fw(b) Pend(8px) Bdendc($bd-primary-divider) Bdends(s) Bdendw(1px)' ).find('span' ,class_="Fz(12px) C($c-icon)" ).text
  flex['body']['contents'][4]['contents'][0]['contents'][1]['text'] = soup.find('div',class_='D(f) Fld(c) Ai(c) Fw(b) Pend(8px) Bdendc($bd-primary-divider) Bdends(s) Bdendw(1px)' ).find('span' ,class_="Fz(16px) C($c-link-text) Mb(4px)" ).text
  #成交值
  flex['body']['contents'][4]['contents'][1]['contents'][0]['text'] = soup.find('ul',class_='D(f) Fld(c) Flw(w) H(192px) Mx(-16px)' ).find_all('li',class_='price-detail-item H(32px) Mx(16px) D(f) Jc(sb) Ai(c) Bxz(bb) Px(0px) Py(4px) Bdbs(s) Bdbc($bd-primary-divider) Bdbw(1px)')[5].find_all('span')[0].text
  flex['body']['contents'][4]['contents'][1]['contents'][1]['text'] = soup.find('ul',class_='D(f) Fld(c) Flw(w) H(192px) Mx(-16px)' ).find_all('li',class_='price-detail-item H(32px) Mx(16px) D(f) Jc(sb) Ai(c) Bxz(bb) Px(0px) Py(4px) Bdbs(s) Bdbc($bd-primary-divider) Bdbw(1px)')[5].find_all('span')[1].text
  # 內盤
  flex['body']['contents'][4]['contents'][2]['contents'][0]['text'] = soup.find('div',class_='D(f) Jc(sb) Ai(c) Mb(4px) Fz(16px)--mobile Fz(14px)' ).find_all('div',class_='C(#232a31) Fw(b)')[0].find_all('span')[0].text
  flex['body']['contents'][4]['contents'][2]['contents'][1]['text'] = soup.find('div',class_='D(f) Jc(sb) Ai(c) Mb(4px) Fz(16px)--mobile Fz(14px)' ).find_all('div',class_='C(#232a31) Fw(b)')[0].find_all('span')[1].text
  # 外盤
  flex['body']['contents'][4]['contents'][3]['contents'][0]['text'] = soup.find('div',class_='D(f) Jc(sb) Ai(c) Mb(4px) Fz(16px)--mobile Fz(14px)' ).find_all('div',class_='C(#232a31) Fw(b)')[1].find_all('span')[2].text
  flex['body']['contents'][4]['contents'][3]['contents'][1]['text'] = soup.find('div',class_='D(f) Jc(sb) Ai(c) Mb(4px) Fz(16px)--mobile Fz(14px)' ).find_all('div',class_='C(#232a31) Fw(b)')[1].find_all('span')[0].text
  #本益比
  flex['body']['contents'][4]['contents'][4]['contents'][0]['text'] = soup.find('div',class_='D(f) Fld(c) Ai(c) Fw(b) Px(8px) Bdendc($bd-primary-divider) Bdends(s) Bdendw(1px)' ).find('span',class_='Fz(12px) C($c-icon)').text  
  flex['body']['contents'][4]['contents'][4]['contents'][1]['text'] = soup.find('div',class_='D(f) Fld(c) Ai(c) Fw(b) Px(8px) Bdendc($bd-primary-divider) Bdends(s) Bdendw(1px)' ).find('span',class_='Fz(16px) C($c-link-text) Mb(4px)').text 
  # 個股新聞
  url = "https://tw.stock.yahoo.com/quote/"+ record[1] +"/news"
  a= requests.get(url ,headers =headers , allow_redirects= True)
  a.encoding='utf-8'
  soup = BeautifulSoup(a.text , 'html.parser' )

  flex['body']['contents'][6]['action']['label'] = soup.find_all('div', class_='Ov(h) Pend(14%) Pend(44px)--sm1024')[0].find('h3', class_='Mt(0) Mb(8px)').text
  flex['body']['contents'][6]['action']['uri']  = soup.find_all('div', class_='Ov(h) Pend(14%) Pend(44px)--sm1024')[0].find('a').get('href')

  flex['body']['contents'][7]['action']['label'] = soup.find_all('div', class_='Ov(h) Pend(14%) Pend(44px)--sm1024')[1].find('h3', class_='Mt(0) Mb(8px)').text
  flex['body']['contents'][7]['action']['uri']  = soup.find_all('div', class_='Ov(h) Pend(14%) Pend(44px)--sm1024')[1].find('a').get('href')

  flex['body']['contents'][8]['action']['label'] = soup.find_all('div', class_='Ov(h) Pend(14%) Pend(44px)--sm1024')[2].find('h3', class_='Mt(0) Mb(8px)').text
  flex['body']['contents'][8]['action']['uri']  = soup.find_all('div', class_='Ov(h) Pend(14%) Pend(44px)--sm1024')[2].find('a').get('href')

  flex['body']['contents'][9]['action']['label'] = soup.find_all('div', class_='Ov(h) Pend(14%) Pend(44px)--sm1024')[3].find('h3', class_='Mt(0) Mb(8px)').text
  flex['body']['contents'][9]['action']['uri']  = soup.find_all('div', class_='Ov(h) Pend(14%) Pend(44px)--sm1024')[3].find('a').get('href')

  flex['body']['contents'][10]['action']['uri'] = url

  return flex







# rich menu 功能選單設置 
richmenu_1 = RichMenu(
    size=RichMenuSize(width=2500, height=1686),
    selected=False,
    name="richmenu_1",
    chat_bar_text="功能選單",
    areas=[RichMenuArea(
        bounds=RichMenuBounds(x=0, y=0, width=2500/3, height=843),
        action=PostbackAction(label='即時查詢', data='即時查詢',text='即時查詢')
        ),
        RichMenuArea(
        bounds=RichMenuBounds(x=2500/3, y=0, width=2500/3, height=843),
        action=PostbackAction(label='歷史資料', data='歷史資料',text='歷史資料')
        ),
        RichMenuArea(
        bounds=RichMenuBounds(x=2500*2/3, y=0, width=2500/3, height=843),
        action=PostbackAction(label='技術分析', data='技術分析',text='技術分析')
        ),
        RichMenuArea(
        bounds=RichMenuBounds(x=0, y=843, width=2500/3, height=843),
        action=PostbackAction(label='機器學習預測', data='機器學習預測',text='機器學習預測')
        ),
        RichMenuArea(
        bounds=RichMenuBounds(x=2500/3, y=843, width=2500/3, height=843),
        action=PostbackAction(label='最新消息', data='最新消息',text='最新消息')
        ),
        RichMenuArea(
        bounds=RichMenuBounds(x=2500*2/3, y=843, width=2500/3, height=843),
        action=URIAction(label='意見回饋',uri='https://docs.google.com/forms/d/e/1FAIpQLSeU32QbrDZPIV_BGRypNZwCOdz6Jz8_-zBystvgEzXLMpdiwQ/viewform?usp=sf_link')
        )
    ]
)
rich_menu_id = line_bot_api.create_rich_menu(rich_menu=richmenu_1)

with open(    os.path.join( os.path.dirname(__file__) ,"richmenu_1.png" )  , 'rb') as f:
    line_bot_api.set_rich_menu_image( rich_menu_id, 'image/png', f)

line_bot_api.set_default_rich_menu(rich_menu_id)

def access_database():    
    DATABASE_URL = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cursor = conn.cursor()
    return conn, cursor

####  CallDatabase  TABLE_NAME(a string) replace user_dualtone_settings
def init_table(TABLE_NAME):
    conn, cursor = access_database()
    postgres_table_query = "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema'"
    cursor.execute(postgres_table_query)
    table_records = cursor.fetchall()
    table_records = [i[0] for i in table_records]

    if TABLE_NAME not in table_records:

        create_table_query = """CREATE TABLE """+ TABLE_NAME +""" (
            user_id VARCHAR ( 50 ) PRIMARY KEY,
            problem VARCHAR ( 20 ) NOT NULL,
            stock VARCHAR ( 20 ) NOT NULL,
            period VARCHAR ( 20 ) NOT NULL,
            interval VARCHAR ( 20 ) NOT NULL,
            indicator VARCHAR ( 20 ) NOT NULL,
            model VARCHAR ( 20 ) NOT NULL,
            result_model VARCHAR ( 50 ) NOT NULL,
            predicted_price VARCHAR ( 50 ) NOT NULL
        );"""
        cursor.execute(create_table_query)
        conn.commit()
    return True

def drop_table(TABLE_NAME):
  conn, cursor = access_database()
  delete_table_query = '''DROP TABLE IF EXISTS ''' + TABLE_NAME
  cursor.execute(delete_table_query)
  conn.commit()
  cursor.close()
  conn.close()
  return True


def init_record(user_id,   problem  ,TABLE_NAME ):
    conn, cursor = access_database()
    table_columns = '(user_id,  problem ,stock, period, interval, indicator ,model, result_model , predicted_price)'
    postgres_insert_query = "INSERT INTO "+ TABLE_NAME + f" {table_columns} VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    record = (user_id, problem ,'2330.TW', '3y', '1d', 'MACD','LSTM' ,'0','0')
    cursor.execute(postgres_insert_query, record)
    conn.commit()
    cursor.close()
    conn.close()
    return record

def check_record(user_id, TABLE_NAME):
    conn, cursor = access_database()
    postgres_select_query = "SELECT * FROM "+ TABLE_NAME + f" WHERE user_id = '{user_id}';"
    cursor.execute(postgres_select_query)
    user_settings = cursor.fetchone()
    return user_settings

def find_record(user_id, TABLE_NAME, col_name):
    conn, cursor = access_database()
    postgres_select_query = "SELECT "+col_name+" FROM "+ TABLE_NAME + f" WHERE user_id = '{user_id}';"
    cursor.execute(postgres_select_query)
    user_settings = cursor.fetchone()
    return user_settings

def update_record(user_id, col, value, TABLE_NAME):
    conn, cursor = access_database()
    postgres_update_query = "UPDATE " + TABLE_NAME +f" SET {col} = %s WHERE user_id = %s"
    cursor.execute(postgres_update_query, (value, user_id))
    conn.commit()
    postgres_select_query = "SELECT * FROM "+ TABLE_NAME + f" WHERE user_id = '{user_id}';"
    cursor.execute(postgres_select_query)
    user_settings = cursor.fetchone()
    cursor.close()
    conn.close()
    return user_settings


def phase_start(event,   problem  , TABLE_NAME):
    # 初始化表格
    init_table(TABLE_NAME)

    # 檢查使用者資料是否存在
    if check_record(event.source.user_id , TABLE_NAME ):
        _ = update_record(event.source.user_id, "problem" , problem , TABLE_NAME)
    else:
        _ = init_record(event.source.user_id,   problem  , TABLE_NAME )  
    line_bot_api.reply_message( event.reply_token, TextSendMessage(text="請輸入股票代碼")   )




def phase_intermediate(event , TABLE_NAME ):
  
    problem_type = find_record(event.source.user_id, TABLE_NAME, "problem")  
    if "即時查詢" in problem_type :
      if event.type=="message":
        update_record(event.source.user_id, "stock", event.message.text , TABLE_NAME )
        record = find_record(event.source.user_id, TABLE_NAME, "problem ,stock")    
        # line_bot_api.reply_message(event.reply_token,TextSendMessage(text=str(record)))
        flex_message = FlexSendMessage( alt_text='行銷搬進大程式', contents= stock_info(record) )
        line_bot_api.reply_message(event.reply_token, flex_message)




    if "歷史資料" in problem_type :
      if event.type=="message":
        update_record(event.source.user_id, "stock", event.message.text , TABLE_NAME )
        mode_dict = {'1d':'1天','5d':'5天','1mo':'1個月','3mo':'3個月','6mo':'6個月','1y':'1年','3y':'3年','5y':'5年','10y':'10年'}
        line_bot_api.reply_message(
          event.reply_token,
            TextSendMessage(
                text=f"請選擇日期範圍", 
                quick_reply=QuickReply(
                    items=[QuickReplyButton(action=PostbackAction(label=v, display_text=f'日期範圍：{v}',data=f'period={k}')) for k, v in mode_dict.items() ]
                )
            )
        )
      if event.type=="postback" and event.postback.data.split('=')[0]=="period":
        record = update_record(event.source.user_id, event.postback.data.split('=')[0] , event.postback.data.split('=')[1] , TABLE_NAME )
        mode_dict = {'1m':'1分','15m':'15分','60m':'60分','90m':'90分','1h':'1小時','1d':'1天','5d':'5天','1wk':'1週','1mo':'1個月','3mo':'3個月'}
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text=f"請選擇數據頻率", 
                quick_reply=QuickReply(
                    items=[QuickReplyButton(action=PostbackAction(label=v, display_text=f'數據頻率：{v}',data=f'interval={k}')) for k, v in mode_dict.items() ]
                )
            )
        )      
      if event.type=="postback" and event.postback.data.split('=')[0]=="interval":
        update_record(event.source.user_id, event.postback.data.split('=')[0] , event.postback.data.split('=')[1] , TABLE_NAME )
        record = find_record(event.source.user_id, TABLE_NAME, "problem ,stock, period, interval")    
        line_bot_api.reply_message(event.reply_token,TextSendMessage(text=str(record)))




    if "技術分析" in problem_type :
      if event.type=="message":
        update_record(event.source.user_id, "stock", event.message.text , TABLE_NAME )
        mode_dict = {'1d':'1天','5d':'5天','1mo':'1個月','3mo':'3個月','6mo':'6個月','1y':'1年','3y':'3年','5y':'5年','10y':'10年'}
        line_bot_api.reply_message(
          event.reply_token,
            TextSendMessage(
                text=f"請選擇日期範圍", 
                quick_reply=QuickReply(
                    items=[QuickReplyButton(action=PostbackAction(label=v, display_text=f'日期範圍：{v}',data=f'period={k}')) for k, v in mode_dict.items() ]
                )
            )
        )
      if event.type=="postback" and event.postback.data.split('=')[0]=="period":
        record = update_record(event.source.user_id, event.postback.data.split('=')[0] , event.postback.data.split('=')[1] , TABLE_NAME )
        mode_dict = {'1m':'1分','15m':'15分','60m':'60分','90m':'90分','1h':'1小時','1d':'1天','5d':'5天','1wk':'1週','1mo':'1個月','3mo':'3個月'}
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text=f"請選擇數據頻率", 
                quick_reply=QuickReply(
                    items=[QuickReplyButton(action=PostbackAction(label=v, display_text=f'數據頻率：{v}',data=f'interval={k}')) for k, v in mode_dict.items() ]
                )
            )
        )    
      if event.type=="postback" and event.postback.data.split('=')[0]=="interval":
        record = update_record(event.source.user_id, event.postback.data.split('=')[0] , event.postback.data.split('=')[1] , TABLE_NAME )
        mode_dict = {'MACD':'MACD','KD':'KD','RSI':'RSI'}
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text=f"請選擇指標", 
                quick_reply=QuickReply(
                    items=[QuickReplyButton(action=PostbackAction(label=v, display_text=f'指標：{v}',data=f'indicator={k}')) for k, v in mode_dict.items() ]
                )
            )
        )
      if event.type=="postback" and event.postback.data.split('=')[0]=="indicator":

        update_record(event.source.user_id, event.postback.data.split('=')[0] , event.postback.data.split('=')[1] , TABLE_NAME )
        record = find_record(event.source.user_id, TABLE_NAME, "problem ,stock, period, interval, indicator")            

        # line_bot_api.reply_message(event.reply_token,TextSendMessage(text=str(record)))
        image_uri = analysis_plot(record)
        line_bot_api.reply_message( event.reply_token, ImageSendMessage(original_content_url=image_uri, preview_image_url=image_uri)  )
 


    if "機器學習預測" in problem_type :
      if event.type=="message":
        update_record(event.source.user_id, "stock", event.message.text , TABLE_NAME )
        mode_dict = {'LSTM':'LSTM','RF':'RF'}
        line_bot_api.reply_message(
          event.reply_token,
            TextSendMessage(
                text=f"請選擇預測模型\n(選擇後需要重新訓練,請稍等)", 
                quick_reply=QuickReply(
                    items=[QuickReplyButton(action=PostbackAction(label=v, display_text=f'預測模型：{v}',data=f'model={k}')) for k, v in mode_dict.items() ]
                )
            )
        )
      if event.type=="postback" and event.postback.data.split('=')[0]=="model":
        update_record(event.source.user_id, event.postback.data.split('=')[0] , event.postback.data.split('=')[1] , TABLE_NAME )
        record = find_record(event.source.user_id, TABLE_NAME, "problem ,stock, model") 
        if record[2]=="LSTM":
          predicted_stock_price1 , img_uri , model_summary = LSTM_model(record)
          out=[]
          out.append( TextSendMessage(text= "以下為模型"+record[2]+"預測"+ record[1]  + "的結果" ) )                
          out.append( ImageSendMessage(original_content_url=img_uri, preview_image_url=img_uri) )
          out.append( TextSendMessage(text= "預測價格為: "+ str(predicted_stock_price1[0][0]) ) )
          line_bot_api.reply_message(event.reply_token,out )
        if record[2]=="RF":
          out = RF_model(record)
          line_bot_api.reply_message(event.reply_token, TextSendMessage(text= out )  )
            

# 文字事件
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message = TextSendMessage(text=event.message.text)
  
    if len(get_stock(event.message.text ,'3y' ,'1d' ))!=0  :
      phase_intermediate(event, 'your_table')
    if len(get_stock(event.message.text ,'3y' ,'1d' ))==0 and event.message.text not in ["即時查詢" , "歷史資料","技術分析" ,"即時查詢"  ,"機器學習預測" , "最新消息"]:
      line_bot_api.reply_message(event.reply_token, TextSendMessage(text= "找不到您所需的資料，更多代號請上yahoo奇摩股市查詢" )  )
        
   

# postback event事件
@handler.add(PostbackEvent)
def handle_postback(event):

    if event.postback.data=="即時查詢" :     
      phase_start(event,"即時查詢" ,  'your_table' )
    if event.postback.data=="歷史資料" :
      line_bot_api.reply_message(event.reply_token, TextSendMessage(text= "功能開發中,敬請期待" )  )
#       phase_start(event,"歷史資料" ,  'your_table' )
    if event.postback.data=="技術分析" :     
      phase_start(event,"技術分析" ,  'your_table' )
    if event.postback.data=="即時查詢" :     
      phase_start(event,"即時查詢" ,  'your_table' )
    if event.postback.data=="機器學習預測" :     
       phase_start(event,"機器學習預測" ,  'your_table' ) 
    if event.postback.data=="最新消息" :     
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text= "功能開發中,敬請期待" ) )
        
    if event.postback.data.startswith('period=') or event.postback.data.startswith('interval=') or event.postback.data.startswith('indicator=') or event.postback.data.startswith('model='):
      phase_intermediate(event, 'your_table')




#主程式
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
