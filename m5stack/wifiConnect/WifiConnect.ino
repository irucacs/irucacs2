#include <M5Stack.h>
#include <WiFi.h>

#define WIFI_SSID "DolphinLAN24"
#define WIFI_PASSWORD "dolphin21"

void setup() {
  M5.begin();

  // Wi-Fi接続
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("connecting");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
    M5.Lcd.println("Connecting");
  }
  Serial.println();

  // WiFi Connected
  Serial.println("\nWiFi Connected.");
  Serial.println(WiFi.localIP());
  M5.Lcd.setTextSize(3);
  M5.Lcd.println("WiFi Connected:");
  M5.Lcd.println(WiFi.localIP());
  M5.Lcd.println("");

}

void loop() {

}
