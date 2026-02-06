
/*************************************************
 * Nano 33 BLE Rev2
 * 2x CD4051 + 10x FSR + IMU (Accel + Gyro + Mag)
 * 单行 CSV 输出版本
 *************************************************/

#include <Arduino_BMI270_BMM150.h>

// ---------- CD4051 控制引脚 ----------
const int PIN_S0 = 2;
const int PIN_S1 = 3;
const int PIN_S2 = 4;

// ---------- CD4051 信号引脚 ----------
const int PIN_SIG_MUX1 = A0;
const int PIN_SIG_MUX2 = A1;

// ---------- 参数 ----------
const int SETTLE_US = 5;
const int AVG_COUNT = 5;
const int FSR_COUNT_MUX1 = 8;
const int FSR_COUNT_MUX2 = 2;

// ---------- IMU ----------
const unsigned long IMU_UPDATE_INTERVAL = 50;
unsigned long lastImuUpdate = 0;

// ---------- 数据 ----------
int fsr[10];               // FSR1~FSR10
float ax, ay, az;
float gx, gy, gz;
float mx, my, mz;

// ------------------------------------------------

void setup() {
  Serial.begin(115200);
  while (!Serial);

  pinMode(PIN_S0, OUTPUT);
  pinMode(PIN_S1, OUTPUT);
  pinMode(PIN_S2, OUTPUT);

  digitalWrite(PIN_S0, LOW);
  digitalWrite(PIN_S1, LOW);
  digitalWrite(PIN_S2, LOW);

  analogReadResolution(12);

  if (!IMU.begin()) {
    while (1);
  }

  // 可选：打印 CSV 表头（只打印一次）
  Serial.println(
    "time,"
    "FSR1,FSR2,FSR3,FSR4,FSR5,FSR6,FSR7,FSR8,FSR9,FSR10,"
    "ax,ay,az,gx,gy,gz,mx,my,mz"
  );
}

// ------------------------------------------------

void selectChannel(uint8_t ch) {
  digitalWrite(PIN_S0, ch & 0x01);
  digitalWrite(PIN_S1, (ch >> 1) & 0x01);
  digitalWrite(PIN_S2, (ch >> 2) & 0x01);
}

int readMuxAnalog(int pin, uint8_t channel) {
  selectChannel(channel);
  delayMicroseconds(SETTLE_US);

  analogRead(pin);
  delayMicroseconds(SETTLE_US);

  long sum = 0;
  for (int i = 0; i < AVG_COUNT; i++) {
    sum += analogRead(pin);
    delayMicroseconds(SETTLE_US);
  }
  return sum / AVG_COUNT;
}

// ------------------------------------------------

void readIMU() {
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);
  }
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gx, gy, gz);
  }
  if (IMU.magneticFieldAvailable()) {
    IMU.readMagneticField(mx, my, mz);
  }
}

// ------------------------------------------------

void loop() {
  // -------- 读取 10 个 FSR --------
  for (int i = 0; i < 8; i++) {
    fsr[i] = readMuxAnalog(PIN_SIG_MUX1, i);
  }
  for (int i = 0; i < 2; i++) {
    fsr[i + 8] = readMuxAnalog(PIN_SIG_MUX2, i);
  }

  // -------- IMU --------
  if (millis() - lastImuUpdate >= IMU_UPDATE_INTERVAL) {
    lastImuUpdate = millis();
    readIMU();

    // -------- 单行 CSV 输出 --------
    Serial.print(millis()); Serial.print(",");

    // FSR 1~10
    for (int i = 0; i < 10; i++) {
      Serial.print(fsr[i]);
      Serial.print(",");
    }

    // IMU 9 轴
    Serial.print(ax, 4); Serial.print(",");
    Serial.print(ay, 4); Serial.print(",");
    Serial.print(az, 4); Serial.print(",");
    Serial.print(gx, 4); Serial.print(",");
    Serial.print(gy, 4); Serial.print(",");
    Serial.print(gz, 4); Serial.print(",");
    Serial.print(mx, 2); Serial.print(",");
    Serial.print(my, 2); Serial.print(",");
    Serial.println(mz, 2);
  }

  delay(10);
}

