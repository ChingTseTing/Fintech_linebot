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



app = Flask(__name__)

# 必須放上自己的Channel Access Token
line_bot_api = LineBotApi('kiO7NFLfre4p/DYSzSC3tO9TcYfkX/W8z5y3l2NI0EVG4w1bGQnG25fgX6KVGLWvTcRIB1qAwhNz2Qkb8pkJDsHtI3A8enXRaSKTZ5kTOR7f5+NWAV1G4MlhTFxULYiSW6XZd+G49DvQhDpeiqrmSwdB04t89/1O/w1cDnyilFU=')
# 必須放上自己的Channel Secret
handler = WebhookHandler('87723191543af71cbeb3eb10170ba058')
# line_bot_api.push_message('U98f95fa0a1fd644fd3c8ce928f9f1eb4', TextSendMessage(text='你可以開始了'))


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


# 文字事件
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message = TextSendMessage(text=event.message.text)

    try: 
      tmp = get_stock( event.message.text )
      if len(tmp)!=0:

        output = [] 
        image_message_url = stock_plot( tmp['Close'] , event.message.text)
        image_message = ImageSendMessage(original_content_url=image_message_url ,preview_image_url = image_message_url)
        output.append(image_message)


        

        carousel_template_message = TemplateSendMessage(
            alt_text='免費教學影片',
            template=CarouselTemplate(
                columns=[
                    CarouselColumn(
                        thumbnail_image_url='https://i.imgur.com/wpM584d.jpg',
                        title='Python基礎教學',
                        text='萬丈高樓平地起',
                        actions=[
                            MessageAction(
                                label='教學內容',
                                text='教學內容'
                            ),
                            URIAction(
                                label='馬上查看',
                                uri='https://marketingliveincode.com/?page_id=270'
                            ),
                            PostbackAction(label='ping with text', data='ping1', text='ping2')
                        ]
                    ),
                    CarouselColumn(
                        thumbnail_image_url='https://i.imgur.com/W7nI6fg.jpg',
                        title='Line Bot聊天機器人',
                        text='台灣最廣泛使用的通訊軟體',
                        actions=[
                            MessageAction(
                                label='教學內容',
                                text='Line Bot申請與串接'
                            ),
                            URIAction(
                                label='馬上查看',
                                uri='https://marketingliveincode.com/?page_id=2532'
                            )
                        ]
                    ),
                    CarouselColumn(
                        thumbnail_image_url='https://i.imgur.com/l7rzfIK.jpg',
                        title='Telegram Bot聊天機器人',
                        text='唯有真正的方便，能帶來意想不到的價值',
                        actions=[
                            MessageAction(
                                label='教學內容',
                                text='Telegrame申請與串接'
                            ),
                            URIAction(
                                label='馬上查看',
                                uri='https://marketingliveincode.com/?page_id=2648'
                            )
                        ]
                    )
                ]
            )
        )


        output.append(carousel_template_message)

        line_bot_api.reply_message(event.reply_token, output)

    except:
      line_bot_api.reply_message(event.reply_token,TextSendMessage(text="額..我找不到"))

# 加入好友事件
@handler.add(FollowEvent)
def handle_follow(event):
    line_bot_api.reply_message( event.reply_token,TextSendMessage(text="你好")  )

  
# postback事件
@handler.add(PostbackEvent)
def handle_postback(event):
    line_bot_api.reply_message( event.reply_token,TextSendMessage(text=  event.postback.data )  )




#主程式
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
