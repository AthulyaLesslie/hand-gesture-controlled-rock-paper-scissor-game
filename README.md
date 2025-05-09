# Hand Gesture Controlled Rock-Paper-Scissors Game

## Project Overview

This is a **Hand Gesture Controlled Rock-Paper-Scissors Game** where players can interact with the game using hand gestures detected via a webcam. The game recognizes three hand gestures:

- ✊ Rock  
- 🖐️ Paper  
- ✌️ Scissors  

The player shows a gesture using their hand in front of the webcam, and the game detects the gesture and plays against the computer. The computer's choice is determined by a random selection of gestures.

This project uses **OpenCV** and **MediaPipe** for real-time hand gesture detection, and **Streamlit** for creating the web app interface. The hand gestures are classified using a **VGG-16 based CNN model** that has been trained on gesture images to recognize different hand poses.

## Features

- Real-time hand gesture recognition using the webcam.
- Interaction with the game using Rock, Paper, and Scissors hand gestures.
- Simple and interactive interface with Streamlit.
- Randomized game responses from the computer.

## Technologies Used

- **OpenCV**: For webcam capture and image processing.
- **MediaPipe**: For hand gesture detection.
- **TensorFlow**: For building and training a VGG-16 based CNN model for gesture recognition.
- **Streamlit**: For creating the web app interface.

## How It Works

1. **Webcam Activation**: When you click on "Start Game," your webcam will open, and you’ll be asked to show a hand gesture.
2. **Gesture Detection**: The webcam feed is processed in real-time with **MediaPipe** to detect the **Rock** gesture. For the **Paper** and **Scissors** gestures, the hand image is passed through a **VGG-16 based CNN model** that has been trained to recognize these gestures.
3. **Game Round**: You will have 2 seconds to show your gesture. The game compares your gesture with the computer's randomly selected gesture to determine the winner of the round.
4. **Game End**: The game continues until either the player or the computer scores 5 points. The player who reaches 5 points wins the game.
