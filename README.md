# FINTECH PROJECT  你今天 line 股了嗎
## 1.檔案說明
 檔案|說明   
  ------------------------ | ---------------------------  
 README                     |   說明文件   
 fintech_final_project.py   |  主程式所在 
 Procfile                   |  告訴 Heroku 我們的應用程式是哪種類型，以及需要執行哪個檔案<br /> 以LINEbot來說，是 web類型應用程式，project 中，需要執行 fintech_final_project.py這個主程式
 requirements.txt           | 告訴 Heroku 提供的伺服器需要安裝那些pytohn套件
 richmenu_1.png             | 客製化的設定機器人聊天室介面功能選單<br />除了在主程式裡設定按鈕在圖片上的相對位置及觸發事件,也要提供背景圖片

## 2.環境搭建
### 2.1 從 [Line Developer](https://account.line.biz/login?redirectUri=https%3A%2F%2Fdevelopers.line.biz%2Fconsole%2Fprovider%2F1656718980%3Fstatus%3Dsuccess%26status%3Dcancelled) 為你的機器人註冊一個line官方帳號
   * Channel access token, Channel secret : API金鑰,是連接fintech_final_project.py跟機器人帳戶的關鍵<br />
                                          安全性考量,兩個API金鑰並未寫死在fintech_final_project.py,而是放在機器人部屬在heroku
                                          伺服器的環境變數空間
   * webhook URL                          : enable, 填入 https://你在heroku建立的伺服器名稱.herokuapp.com/callback
   * 機器人的官方帳號其他主頁設定可在line official account manager 設定

### 2.2 註冊一個 [Heroku](https://www.heroku.com/) 免費帳戶,同時在你的帳戶中新增一個伺服器
   * Heroku 是一個平台即服務(PaaS)平台, 提供的伺服器才是我們的應用程式(linebot)真正運行的地方, 
   * 如果說 linebot 機器人是工人, 伺服器是就是工廠, 要運作就要透過"部屬"這個動作完成(heroku會去抓在github上的code)
   * 電腦端下載heroku CLI, 透過指令,可以隨時監控後台每次收發訊息的狀態
   
### 2.3 連接Heroku Postgres資料庫
   * 是 Heroku 提供的一種PostgreSQL資料庫, 我們額外建立連結到heroku運作的伺服器，用來暫存使用者輸入的設定
   * 直接在 Heroku 新增的伺服器中連結

### 2.4 架構圖
&nbsp;**______**&nbsp;&ensp;&emsp;&emsp;&emsp;&emsp;**________________** &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**____________**<br />
| &emsp;&emsp; |<---------| &nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;|<---- 依據 webhook 事件 透過 LINE Platform 回應用戶 -------|&emsp;&emsp;&emsp;&emsp;&emsp;|<br />
| User | &emsp;&emsp;&emsp;&emsp;| LINE Platform | &nbsp;&nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; | Bot server |<br />
|**______**|--------->|**________________**|------ webhook 事件傳送至 bot server的 webhook URL ----->|**____________**| <br />

&emsp;&emsp;&emsp;&emsp;LINE官方用來傳遞"user"及&emsp;&emsp;&emsp;&emsp;server,server之間是根據LINE官方所提供的&emsp;&emsp;&emsp;開發LINEBot的server <br />
&emsp;&emsp;&emsp;&emsp;"Bot server"資訊的中介server&emsp;&emsp;&emsp;Messaging API 協議(HTTPS協定) 進行溝通


## 3.使用方式
   * Line 好友連結 (LINE id : @074vyxct)  <br />
      ![Imgur](https://i.imgur.com/DKzKNtym.png)
   * demo 影片連結: 
      https://drive.google.com/drive/folders/1dFTDaoglcOdrP-IqejkzuRyPFWAcXgnL?usp=sharing

## 4.参考文章
* https://marketingliveincode.com/?page_id=2532
* https://line-bot-sdk-python.readthedocs.io/en/stable/index.html
* https://ithelp.ithome.com.tw/users/20120178/ironman/2654
* https://ithelp.ithome.com.tw/users/20120178/articles
