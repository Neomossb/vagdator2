#include <Servo.h>

String serialData;

Servo servoA;
Servo servoB;
Servo servoC;
Servo servoD;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.setTimeout(10);

  servoA.attach(8);
  servoB.attach(9);
  servoC.attach(10);
  servoD.attach(11);

  servoA.write(0);
  servoB.write(0);
  servoC.write(0);
  servoD.write(0);
}

void loop() {
  // put your main code here, to run repeatedly:

}

void serialEvent() {
  serialData = Serial.readString();

  servoA.write(parseDataA(serialData));
  servoB.write(parseDataB(serialData));
  servoC.write(parseDataC(serialData));
  servoD.write(parseDataD(serialData));
}

int parseDataA(String data) {
  data.remove(0, data.indexOf("A") + 1);
  data.remove(data.indexOf("B"));

  return data.toInt();
}

int parseDataB(String data) {
  data.remove(0, data.indexOf("B") + 1);
  data.remove(data.indexOf("C"));

  return data.toInt();
}

int parseDataC(String data) {
  data.remove(0, data.indexOf("C") + 1);
  data.remove(data.indexOf("D"));

  return data.toInt();
}

int parseDataD(String data) {
  data.remove(0, data.indexOf("D") + 1);

  return data.toInt();
}

