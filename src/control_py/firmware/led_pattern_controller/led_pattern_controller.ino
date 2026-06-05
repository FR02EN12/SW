const int ld1 = 2;
const int ld2 = 3;
const int ld3 = 4;
const int ld4 = 5;
const int ld5 = 6;

char rawMask[6] = {'0', '0', '0', '0', '0', '\0'};
int rawIndex = -1;

void setup() {
  pinMode(ld1, OUTPUT);
  pinMode(ld2, OUTPUT);
  pinMode(ld3, OUTPUT);
  pinMode(ld4, OUTPUT);
  pinMode(ld5, OUTPUT);

  Serial.begin(9600);
  allOff();

  Serial.println("LED Pattern Controller Ready");
  Serial.println("0 : normal");
  Serial.println("1 : car detected");
  Serial.println("2 : game");
  Serial.println("3 : no game");
  Serial.println("4 : rock");
  Serial.println("5 : scissor");
  Serial.println("6 : paper");
  Serial.println("Mxxxxx : raw 5-bit mask");
}

void loop() {
  if (Serial.available() <= 0) {
    return;
  }

  char cmd = Serial.read();
  if (cmd == '\n' || cmd == '\r') {
    return;
  }

  if (rawIndex >= 0) {
    readRawMaskBit(cmd);
    return;
  }

  if (cmd == 'M' || cmd == 'm') {
    rawIndex = 0;
    return;
  }

  applyPatternCommand(cmd);
}

void readRawMaskBit(char bit) {
  if (bit != '0' && bit != '1') {
    rawIndex = -1;
    Serial.println("Invalid raw mask. Use M followed by five 0/1 bits.");
    return;
  }

  rawMask[rawIndex] = bit;
  rawIndex++;

  if (rawIndex < 5) {
    return;
  }

  rawMask[5] = '\0';
  applyMask(rawMask);
  Serial.print("Label : raw mask ");
  Serial.println(rawMask);
  rawIndex = -1;
}

void applyPatternCommand(char cmd) {
  if (cmd == '0') {
    allOff();
    Serial.println("Label : normal");
  }
  else if (cmd == '1') {
    setLeds(true, true, true, true, true);
    Serial.println("Label : car detected");
  }
  else if (cmd == '2') {
    setLeds(true, false, true, false, true);
    Serial.println("Label : game");
  }
  else if (cmd == '3') {
    setLeds(false, true, false, true, false);
    Serial.println("Label : no game");
  }
  else if (cmd == '4') {
    setLeds(true, true, false, false, false);
    Serial.println("Label : rock");
  }
  else if (cmd == '5') {
    setLeds(true, false, true, false, false);
    Serial.println("Label : scissor");
  }
  else if (cmd == '6') {
    setLeds(true, false, false, true, false);
    Serial.println("Label : paper");
  }
  else {
    Serial.println("Invalid input. Send 0-6 or Mxxxxx.");
  }
}

void applyMask(const char mask[6]) {
  setLeds(
    mask[0] == '1',
    mask[1] == '1',
    mask[2] == '1',
    mask[3] == '1',
    mask[4] == '1'
  );
}

void setLeds(bool b1, bool b2, bool b3, bool b4, bool b5) {
  digitalWrite(ld1, b1 ? HIGH : LOW);
  digitalWrite(ld2, b2 ? HIGH : LOW);
  digitalWrite(ld3, b3 ? HIGH : LOW);
  digitalWrite(ld4, b4 ? HIGH : LOW);
  digitalWrite(ld5, b5 ? HIGH : LOW);
}

void allOff() {
  setLeds(false, false, false, false, false);
}
