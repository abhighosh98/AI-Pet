#include <Servo.h>

Servo hservo;//servo for moving in horizontal axis
Servo vservo; // servo for vertical axis

int hpos=90;
int vpos=90;

int a=5;

void setup() {
  Serial.begin(115200);
  hservo.attach(9); //hservo is now on pin9
  vservo.attach(8); //vservo is now on pin8
  hservo.write(hpos);
  vservo.write(vpos);
}

void loop() {
  if (Serial.available()){
    a=Serial.parseInt();
  }
//  a=Serial.parseInt();
  switch (a) {
     case 1:{
      (vpos+15<180) ? vpos+=15 : vpos=180;
      (hpos-15>0) ? hpos-=15 : hpos=0;
      break;
    }
    case 2:{
      (vpos+15<180) ? vpos+=15 : vpos=180;
      
      break;
    }
    case 3:{
      (hpos+15<180) ? hpos+=15 : hpos=180;
      (vpos+15<180) ? vpos+=15 : vpos=180;
      break;
    }
    case 4:{
      (hpos-15>0) ? hpos-=15 : hpos=0;
      break;
    }
    case 5:{
    break;
    }
    case 6:{
      (hpos+15<180) ? hpos+=15 : hpos=180;
      break;
    }
    case 7:{
      (hpos-15>0) ? hpos-=15 : hpos=0;
      (vpos-15>0) ? vpos-=15 : vpos=0;
      break;
    }
    case 8:{
      (vpos-15>0) ? vpos-=15 : vpos=0;
      break;
    }
    case 9:{
      (hpos+15<180) ? hpos+=15 : hpos=180;
      (vpos-15>0) ? vpos-=15 : vpos=0;
      break;
    }
    default:
    break;
  }
  hservo.write(hpos);
  vservo.write(vpos);
  Serial.print(hpos);
  Serial.print(" ");
  Serial.println(vpos);
  delay(5);
}
