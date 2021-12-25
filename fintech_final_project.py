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


app = Flask(__name__)

# 必須放上自己的Channel Access Token
line_bot_api = LineBotApi('kiO7NFLfre4p/DYSzSC3tO9TcYfkX/W8z5y3l2NI0EVG4w1bGQnG25fgX6KVGLWvTcRIB1qAwhNz2Qkb8pkJDsHtI3A8enXRaSKTZ5kTOR7f5+NWAV1G4MlhTFxULYiSW6XZd+G49DvQhDpeiqrmSwdB04t89/1O/w1cDnyilFU=')
# 必須放上自己的Channel Secret
handler = WebhookHandler('87723191543af71cbeb3eb10170ba058')
# line_bot_api.push_message('U98f95fa0a1fd644fd3c8ce928f9f1eb4', TextSendMessage(text='你可以開始了'))

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
  new_df = get_stock_index( get_stock(record[1] ,'3y' , '1d' )   )
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
  return predicted_stock_price1 ,uploaded_image.link



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
    DATABASE_URL = 'postgres://oslwzkeacbduvb:67563d43dd685b29d24491678f3956baab363d9ad65d1622cc7a8e4472a99940@ec2-34-226-178-146.compute-1.amazonaws.com:5432/d86a5ndsm3tor2'#'heroku config:get DATABASE_URL -a fintech-home23')'
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
            result VARCHAR ( 20 ) NOT NULL
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
    table_columns = '(user_id,  problem ,stock, period, interval, indicator ,model, result)'
    postgres_insert_query = "INSERT INTO "+ TABLE_NAME + f" {table_columns} VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
    record = (user_id, problem ,'2330.TW', '3y', '1d', 'MACD','LSTM' ,'0')
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
        line_bot_api.reply_message(event.reply_token,TextSendMessage(text=str(record)))


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
        predicted_price , img_uri = LSTM_model(record)
        out=[]
        out.append( ImageSendMessage(original_content_url=img_uri, preview_image_url=img_uri) )
        out.append( TextSendMessage(text="預測價格為: "+predicted_price) )        
        line_bot_api.reply_message(event.reply_token,out)


# 文字事件
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message = TextSendMessage(text=event.message.text)
    
    if len(get_stock(event.message.text ,'3y' ,'1d' ))!=0:
      phase_intermediate(event, 'your_table')


# postback event事件
@handler.add(PostbackEvent)
def handle_postback(event):

    if event.postback.data=="即時查詢" :     
      phase_start(event,"即時查詢" ,  'your_table' )
    if event.postback.data=="歷史資料" :     
      phase_start(event,"歷史資料" ,  'your_table' )
    if event.postback.data=="技術分析" :     
      phase_start(event,"技術分析" ,  'your_table' )
    if event.postback.data=="即時查詢" :     
      phase_start(event,"即時查詢" ,  'your_table' )
    if event.postback.data=="機器學習預測" :     
      phase_start(event,"機器學習預測" ,  'your_table' ) 
    if event.postback.data.startswith('period=') or event.postback.data.startswith('interval=') or event.postback.data.startswith('indicator=') or event.postback.data.startswith('model='):
      phase_intermediate(event, 'your_table')




#主程式
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
