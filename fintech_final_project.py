
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


import sys, os
import pyimgur
import seaborn as sns
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials as SAC
import psycopg2



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

# 功能模組-畫圖
def stock_plot( dataframe , msg ):
    sns.set_theme()

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Date')
    ax.set_ylabel('price')
    ax.set_title(msg)

    dataframe.plot()
    plt.savefig('send.png')
    CLIENT_ID = "08680019f3643c6"  #"TingChingTse"
    PATH = "send.png"
    im = pyimgur.Imgur(CLIENT_ID)
    uploaded_image = im.upload_image(PATH, title="Uploaded with PyImgur")
    return uploaded_image.link

# 功能模組-求股價
def get_stock(stock_id ):
  tmp = yf.download(stock_id , period='3y',interval='1d' )# start='2016-01-01',end=datetime.now().strftime('%Y-%m-%d')
  return tmp

# rich menu 功能選單設置 
richmenu_1 = RichMenu(
    size=RichMenuSize(width=2500, height=1686),
    selected=False,
    name="richmenu_1",
    chat_bar_text="功能選單",
    areas=[RichMenuArea(
        bounds=RichMenuBounds(x=0, y=0, width=2500/3, height=843),
        action=PostbackAction(label='即時查詢', data='即時查詢建置中', text='即時查詢')
        ),
        RichMenuArea(
        bounds=RichMenuBounds(x=2500/3, y=0, width=2500/3, height=843),
        action=PostbackAction(label='歷史資料', data='歷史資料建置中', text='歷史資料')
        ),
        RichMenuArea(
        bounds=RichMenuBounds(x=2500*2/3, y=0, width=2500/3, height=843),
        action=PostbackAction(label='技術分析', data='技術分析建置中', text='技術分析')
        ),
        RichMenuArea(
        bounds=RichMenuBounds(x=0, y=843, width=2500/3, height=843),
        action=PostbackAction(label='機器學習預測', data='機器學習預測建置中', text='機器學習預測')
        ),
        RichMenuArea(
        bounds=RichMenuBounds(x=2500/3, y=843, width=2500/3, height=843),
        action=PostbackAction(label='最新消息', data='最新消息建置中', text='最新消息')
        ),
        RichMenuArea(
        bounds=RichMenuBounds(x=2500*2/3, y=843, width=2500/3, height=843),
        action=PostbackAction(label='意見回饋', data='意見回饋建置中', text='意見回饋')
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
            message_id VARCHAR ( 50 ) NOT NULL,
            mode VARCHAR ( 20 ) NOT NULL,
            gradient_factor VARCHAR ( 20 ) NOT NULL,
            first_tone VARCHAR ( 20 ) NOT NULL,
            second_tone VARCHAR ( 20 ) NOT NULL
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


def init_record(user_id, message_id, TABLE_NAME):
    conn, cursor = access_database()
    table_columns = '(user_id, message_id, mode, gradient_factor, first_tone, second_tone)'
    postgres_insert_query = "INSERT INTO "+ TABLE_NAME + f" {table_columns} VALUES (%s,%s,%s,%s,%s,%s)"
    record = (user_id, message_id, 'blend', '50', 'red', 'blue')
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

def phase_start(event, TABLE_NAME):
    # 初始化表格
    init_table(TABLE_NAME)

    # 檢查使用者資料是否存在
    if check_record(event.source.user_id , TABLE_NAME ):
        _ = update_record(event.source.user_id, 'message_id', event.message.id , TABLE_NAME)
    else:
        _ = init_record(event.source.user_id, event.message.id , TABLE_NAME)

    mode_dict = {'blend': '線性疊圖', 'composite': '濾鏡疊圖', 'composite_invert': '反式濾鏡疊圖'}
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text=f"[1/4] 今晚，我想來點雙色打光！\n請選擇雙色打光模式：", 
            quick_reply=QuickReply(
                items=[QuickReplyButton(action=PostbackAction(
                    label=v, 
                    display_text=f'打光模式：{v}',
                    data=f'mode={k}')) for k, v in mode_dict.items()
                ]
            )
        )
    )
def phase_intermediate(event , TABLE_NAME ):

    color_dict = {
        'red': '紅',
        'orange': '橙',
        'yellow': '黃',
        'green': '綠',
        'blue': '藍',
        'purple': '紫'
    }
                  
    reply_dict = {
        'mode': '[2/4] 今晚，繼續來點雙色打光！\n請選擇色彩變化梯度：',
        'gradient_factor': '[3/4] 今晚，還想來點雙色打光！\n請選擇第一道色彩：',
        'first_tone': '[4/4] 今晚，最後來點雙色打光！\n請選擇第二道色彩：'
    }
    
    quick_button_dict = {
        'mode': 
        [QuickReplyButton(
            action=PostbackAction(
                label=i, 
                display_text=f'變化梯度：{i}', 
                data=f'gradient_factor={i}')) for i in (5, 10, 50, 100)
        ],
        'gradient_factor': 
        [QuickReplyButton(
            action=PostbackAction(
                label=j, 
                display_text=f'第一道色彩：{j}', 
                data=f'first_tone={i}')) for i, j in color_dict.items()
        ],
        'first_tone':
        [QuickReplyButton(
            action=PostbackAction(
                label=j, 
                display_text=f'第二道色彩：{j}', 
                data=f'second_tone={i}')) for i, j in color_dict.items()
        ]
    }
    
    user_id = event.source.user_id
    postback_data = event.postback.data  #   event.postback.data.split('=')[0] 
    current_phase = postback_data.split('=')[0]

    # 依照使用者的選擇更新資料
    update_record(user_id, current_phase, postback_data.split('=')[1] , TABLE_NAME )
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text=reply_dict[current_phase],
            quick_reply=QuickReply(
                items=quick_button_dict[current_phase]))
        )
    
def phase_finish(event , TABLE_NAME ):
    user_id = event.source.user_id
    postback_data = event.postback.data
    current_phase = postback_data.split('=')[0]

    # 更新資料並取得最後的完整設定
    record = update_record(user_id, current_phase, postback_data.split('=')[1] , TABLE_NAME )

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=str(record))
    )



# 加入好友事件
@handler.add(FollowEvent)
def handle_follow(event):
    line_bot_api.reply_message( event.reply_token,TextSendMessage(text="你好")  )

# 文字事件
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message = TextSendMessage(text=event.message.text)


@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    phase_start(event, 'user_dualtone_settings')

@handler.add(PostbackEvent)
def handle_postback(event):
    if not event.postback.data.startswith('second_tone='): 
        phase_intermediate(event, 'user_dualtone_settings')
    else:
        phase_finish(event, 'user_dualtone_settings')



#主程式
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
