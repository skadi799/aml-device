/*************************************************
 * Nano 33 Rev2 + 2x CD4051 + 10x FSR
 * 模拟多路复用读取（稳定版）
 * CD4051#1: 8个FSR (X0-X7) -> A0
 * CD4051#2: 2个FSR (X0-X1) -> A1
 * 共用控制引脚: S0,S1,S2 (pin 2,3,4)
 *************************************************/

// --------- CD4051 控制引脚 ----------
const int PIN_S0  = 2;
const int PIN_S1  = 3;
const int PIN_S2  = 4;

// --------- CD4051 信号引脚 ----------
const int PIN_SIG_MUX1 = A0;  // 第一个多路复用器（8个FSR）
const int PIN_SIG_MUX2 = A1;  // 第二个多路复用器（2个FSR）

// --------- 参数可调 ----------
const int SETTLE_US = 5;      // 通道切换稳定时间（微秒）
const int AVG_COUNT = 5;      // 每通道取平均次数
const int FSR_COUNT_MUX1 = 8; // 第一个多路复用器FSR数量
const int FSR_COUNT_MUX2 = 2; // 第二个多路复用器FSR数量
const int TOTAL_FSR_COUNT = FSR_COUNT_MUX1 + FSR_COUNT_MUX2; // 总共10个FSR

int fsr_mux1[FSR_COUNT_MUX1]; // 存储第一个多路复用器的FSR读数
int fsr_mux2[FSR_COUNT_MUX2]; // 存储第二个多路复用器的FSR读数

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // 设置控制引脚
  pinMode(PIN_S0, OUTPUT);
  pinMode(PIN_S1, OUTPUT);
  pinMode(PIN_S2, OUTPUT);

  // 初始化控制引脚为低电平
  digitalWrite(PIN_S0, LOW);
  digitalWrite(PIN_S1, LOW);
  digitalWrite(PIN_S2, LOW);

  analogReadResolution(12);  // Nano 33 Rev2 推荐

  Serial.println("2x CD4051 + 10x FSR ready");
  Serial.println("MUX1: FSR1-FSR8 (ch0-7)");
  Serial.println("MUX2: FSR9-FSR10 (ch0-1)");
}

// choose CD4051 channel（0~7）
void selectChannel(uint8_t channel) {
  digitalWrite(PIN_S0, channel & 0x01);
  digitalWrite(PIN_S1, (channel >> 1) & 0x01);
  digitalWrite(PIN_S2, (channel >> 2) & 0x01);
}

// read the analog data of first multiplexer(8 FSR)
int readMux1Analog(uint8_t channel) {
  if (channel >= FSR_COUNT_MUX1) return 0; // security check
  
  selectChannel(channel);
  delayMicroseconds(SETTLE_US);

  // drop first data (ADC residue)
  analogRead(PIN_SIG_MUX1);
  delayMicroseconds(SETTLE_US);

  long sum = 0;
  for (int i = 0; i < AVG_COUNT; i++) {
    sum += analogRead(PIN_SIG_MUX1);
    delayMicroseconds(SETTLE_US);
  }

  return sum / AVG_COUNT;
}

// 读取第二个多路复用器（2个FSR）的模拟值
int readMux2Analog(uint8_t channel) {
  if (channel >= FSR_COUNT_MUX2) return 0; // 安全检查
  
  selectChannel(channel);
  delayMicroseconds(SETTLE_US);

  // 丢弃第一次读数（ADC 残留）
  analogRead(PIN_SIG_MUX2);
  delayMicroseconds(SETTLE_US);

  long sum = 0;
  for (int i = 0; i < AVG_COUNT; i++) {
    sum += analogRead(PIN_SIG_MUX2);
    delayMicroseconds(SETTLE_US);
  }

  return sum / AVG_COUNT;
}

void loop() {
  // 读取第一个多路复用器的8个FSR
  for (int i = 0; i < FSR_COUNT_MUX1; i++) {
    fsr_mux1[i] = readMux1Analog(i);
  }

  // 读取第二个多路复用器的2个FSR
  for (int i = 0; i < FSR_COUNT_MUX2; i++) {
    fsr_mux2[i] = readMux2Analog(i);
  }

  // 串口输出 - 第一个多路复用器 (FSR1-FSR8)
  for (int i = 0; i < FSR_COUNT_MUX1; i++) {
    Serial.print("FSR");
    Serial.print(i + 1);
    Serial.print(": ");
    Serial.print(fsr_mux1[i]);
    Serial.print(" | ");
  }

  // 串口输出 - 第二个多路复用器 (FSR9-FSR10)
  for (int i = 0; i < FSR_COUNT_MUX2; i++) {
    Serial.print("FSR");
    Serial.print(i + 1 + FSR_COUNT_MUX1); // 从9开始编号
    Serial.print(": ");
    Serial.print(fsr_mux2[i]);
    if (i < FSR_COUNT_MUX2 - 1) Serial.print(" | ");
  }
  
  Serial.println();
  delay(50);
}