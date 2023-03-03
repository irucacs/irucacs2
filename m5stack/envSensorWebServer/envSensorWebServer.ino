#include <M5Stack.h>
#include <WiFi.h>
#include "M5_ENV.h"
#include <WebServer.h>

SHT3X sht30;
QMP6988 qmp6988;

//無線LAN情報
#define WIFI_SSID "GS-LAN24"
#define WIFI_PASSWORD "guestGUEST21"

//IPアドレス固定化
const IPAddress ip(10,2,0,23);
const IPAddress gateway(10,2,0,5);
const IPAddress subnet(255,0,0,0);

//センサー情報初期化
float tmp      = 0.0;
float hum      = 0.0;
float pressure = 0.0;

//サーバオブジェクト作成
WebServer server(80);

//関数定義
void handleTemp();
void handleNotFound();

void setup() {
  M5.begin();
  //バックライトの明るさ調整
  M5.Lcd.setBrightness(1);

  // Wi-Fi接続
  WiFi.config(ip, gateway, subnet);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("connecting");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
    M5.Lcd.print("connect..");
  }
  Serial.println();

  // WiFiステータス表示
  M5.Lcd.println();
  Serial.println("\nWiFi Connected.");
  Serial.println(WiFi.localIP());
  M5.Lcd.setTextSize(3);
  M5.Lcd.println("WiFi Connected:");
  M5.Lcd.println(WiFi.localIP());

  //センサー初期化
  Wire.begin();
  qmp6988.init();

  //HTTPサーバー初期化
  // URLにアクセスされた際の動作を登録
  server.on("/", handleTemp);
   // server.onで登録されていないURLにアクセスされた際の動作を登録
  server.onNotFound(handleNotFound);
   // クライアントからの接続応答待ちを開始
  server.begin();
  M5.Lcd.println("HTTP SV started");
  M5.Lcd.println("");
}

void loop() {
  //センサー値取得
  pressure = qmp6988.calcPressure();
  if (sht30.get() == 0) {
    tmp = sht30.cTemp;
    hum = sht30.humidity;
  } else {
    tmp = 0, hum = 0;
  }
  M5.lcd.fillRect(0, 100, 100, 60,BLACK);
  M5.lcd.setCursor(0, 100);
  M5.Lcd.printf("Temp: %2.1f  \r\nHumi: %2.0f%%  \r\nPressure:%2.0fPa\r\n",tmp, hum, pressure);
  delay(2000);
  //HTTP要求待機
  server.handleClient();

}

//HTML温度表示
void handleTemp(){
  char buf[400];
  pressure = qmp6988.calcPressure();
  if (sht30.get() == 0) {
    tmp = sht30.cTemp;
    hum = sht30.humidity;
  } else {
    tmp = 0, hum = 0;
  }
  sprintf(buf, 
    "<html>\
     <head>\
        <title>Server Room Temperature Display</title>\
     </head>\
     <body>\
        <h1>Server Room Temperature Display</h1>\
        <p>Temp: %.2f </p>\
        <p>Humi: %2.0f%% </p>\
        <p>Pressure: %2.0fPa</p>\
     </body>\
     </html>",
  tmp,hum,pressure);
  server.send(200, "text/html", buf);
  M5.Lcd.println("accessed on \"/\"");
}

void handleNotFound(){
  server.send(404, "text/plain", "File Not Found\n\n");
  M5.Lcd.println("accessed 404");
}
