
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
import seaborn as sns
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials as SAC
import psycopg2
import mplfinance as mpf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import talib as ta


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

# 功能模組-畫圖
def analysis_plot(record):
  tt =  get_stock( record[1] ,record[2] , record[3] )

  # Create subplots and mention plot grid size
  fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=( record[1] , ''), row_width=[0.2, 0.2,0.7])
  # Plot OHLC on 1st row
  fig.add_trace(go.Candlestick(x=tt.index, open=tt["Open"], high=tt["High"],low=tt["Low"], close=tt["Close"], name="OHLC", showlegend=True ), row=1, col=1 )
  fig.add_trace(go.Scatter(x=tt.index, y=ta.SMA( np.array(tt['Close']) ,timeperiod=5), mode='lines' ,name='MA5', showlegend=True)  , row=1, col=1 )
  fig.add_trace(go.Scatter(x=tt.index, y=ta.SMA( np.array(tt['Close']) ,timeperiod=10), mode='lines' ,name='MA10', showlegend=True)  , row=1, col=1 )

  # Bar trace for volumes on 2nd row without legend
  fig.add_trace(go.Bar(x=tt.index, y=tt['Volume'], showlegend=False), row=2, col=1)

  fig.add_trace(go.Scatter(x=tt.index, y=ta.RSI( np.array(tt['Close']) ,timeperiod=14), mode='lines' ,name='RSI', showlegend=True)  , row=3, col=1 )


  fig['layout']['yaxis']['title']='Price'
  fig['layout']['yaxis2']['title']='Volume'
  fig['layout']['yaxis3']['title']='RSI'
  # fig.update_layout(

  #     paper_bgcolor="LightSteelBlue",
  # )

  # Do not show OHLC's rangeslider plot 
  fig.update(layout_xaxis_rangeslider_visible=False)
  fig.write_image("send.png")

  CLIENT_ID = "08680019f3643c6"  #"TingChingTse"
  PATH = "send.png"
  im = pyimgur.Imgur(CLIENT_ID)
  uploaded_image = im.upload_image(PATH, title="Uploaded with PyImgur")

  return uploaded_image.link



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
            indicator VARCHAR ( 20 ) NOT NULL
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
    table_columns = '(user_id,  problem ,stock, period, interval, indicator)'
    postgres_insert_query = "INSERT INTO "+ TABLE_NAME + f" {table_columns} VALUES (%s,%s,%s,%s,%s,%s)"
    record = (user_id, problem ,'2330.TW', '3y', '1d', 'MACD')
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


##  AlmaTalks.py

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
        mode_dict = {'MACD':'MACD','RSI':'RSI'}
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
 


# # 加入好友事件
# @handler.add(FollowEvent)
# def handle_follow(event):
#     line_bot_api.reply_message( event.reply_token,TextSendMessage(text="你好")  )

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
    if event.postback.data.startswith('period=') or event.postback.data.startswith('interval=') or event.postback.data.startswith('indicator='):
      phase_intermediate(event, 'your_table')
   
#主程式
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
