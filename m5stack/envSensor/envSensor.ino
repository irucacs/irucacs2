#include <M5Stack.h>
#include "M5_ENV.h"

SHT3X sht30;
QMP6988 qmp6988;

float tmp      = 0.0;
float hum      = 0.0;
float pressure = 0.0;

void setup() {
    M5.begin();
    M5.Power.begin();
    M5.lcd.setTextSize(3);
    Wire.begin();
    qmp6988.init();
}

void loop() {
    pressure = qmp6988.calcPressure();
    if (sht30.get() == 0) {
        tmp = sht30.cTemp;
        hum = sht30.humidity;
    } else {
        tmp = 0, hum = 0;
    }
    M5.lcd.fillRect(0, 20, 100, 60,BLACK);
    M5.lcd.setCursor(0, 20);
    M5.Lcd.printf("Temp: %2.1f  \r\nHumi: %2.0f%%  \r\nPressure:%2.0fPa\r\n",tmp, hum, pressure);
    delay(2000);
}