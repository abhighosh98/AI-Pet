# AI-Pet
This was the code we wrote back in Engineering college for our final year project. As of the date of pushing this to Git it has been 2 years since we have seen the code, so some description may be lost to memory.

The Idea was to create a pet robot that had AI ML capabilites. We first used just a raspberry  pi but then we realised its weaknesses and used a an Arduino to do the heavy electronics work.

The raspbery pi is good for OpenCv but we soon realized that the frames were slow or would overload the Pi. so out architecture became a remote server doing all the heavy ml work, a raspberry pi doing all the on bot work and arduino doing mechanical work.

# Working

Following is the Connection Diagram

![Data connections](https://user-images.githubusercontent.com/61613837/160155450-33876a80-1a64-42f1-a62f-7efb6bfac962.png)


Following is the Architecture Diagram
![Architecture Block](https://user-images.githubusercontent.com/61613837/160155640-6d58a3ee-d64b-4f15-aa79-bcf811c5c516.png)

# Features
1) Detect and track a face (only the head of the bot)
2) Recognise a face
3) Understand emotions
4) Can follow a person wearing Fluorescent Yellow sneakers
5) Play stone paper sissors
6) Can play music and dance

# Note
You dont need a pi cam necessarily, you can use any Ip Camera

# Hardware Used
1) Raspberry Pi 4
2) Remote server (any computer capable or running ML Algos)
3) Arduino boards and assorted modules
4) IP camera: can be a mobile phone cam or even a Pi cam

# Authors
* **Elton Lemos** - [icefrostpeng](https://github.com/icefrostpeng)
* **Abhishek Ghoshal** - [abhighosh98](https://github.com/abhighosh98)
* **Aditya Aspat** - Not on Git yet

# AI-Pet
