
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import random

# Load CNN model for Paper and Scissors
model = load_model('rps_model.h5', compile=False)
cnn_labels = ['Paper', 'Scissors']  # Only Paper and Scissors for CNN

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)
# Check if fingers are folded (for Rock)
def is_rock(hand_landmarks):
    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky fingertips
    folded = 0
    for tip in tips:
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip - 2].y:
            folded += 1
    return folded == 4  # All fingers folded

# Computer's choice logic
def computer_choice():
    return random.choice(['Rock', 'Paper', 'Scissors'])

# Game result logic
def get_winner(player, computer):
    if player == computer:
        return 'Draw'
    elif (player == 'Rock' and computer == 'Scissors') or \
         (player == 'Paper' and computer == 'Rock') or \
         (player == 'Scissors' and computer == 'Paper'):
        return 'Player'
    else:
        return 'Computer'
# Initialize scores
player_score = 0
computer_score = 0

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width to 1280
cap.set(4, 720)   # Set height to 720
start_time = time.time()
warmup_time = 5  # Initial warmup time before the game starts

while True:
    # Countdown before the game starts (2 seconds)
    for i in range(2, 0, -1):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, f"Get Ready! {i}", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
        cv2.imshow("Rock-Paper-Scissors Game", frame)
        if cv2.waitKey(1000) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Start round timer
    round_start = time.time()
    player_choice = "None"
    prediction_text = "Detecting..."
    
    while time.time() - round_start < 2:  # 2-second round timer
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)  # Flip frame horizontally for natural mirror effect
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for MediaPipe
        result = hands.process(img_rgb)

        # Check for hand landmarks
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                if is_rock(handLms):  # Detect Rock using hand landmarks
                    prediction_text = "Rock"
                else:
                    # Use CNN for Paper and Scissors
                    roi = cv2.resize(frame, (224, 224))  
                    roi = roi.astype('float32') / 255.0  
                    roi = np.expand_dims(roi, axis=0)  

                    # Get predictions from the CNN model
                    prediction = model.predict(roi)
                    class_index = np.argmax(prediction) 

                    if class_index == 0:  
                        prediction_text = "Paper"
                    else:  
                        prediction_text = "Scissors"

        # Display the round countdown and prediction
        cv2.putText(frame, f"Detecting... {prediction_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        remaining_time = 2 - int(time.time() - round_start)
        # Move "Time Left" text to a more left position
        cv2.putText(frame, f"Time Left: {remaining_time}s", (900, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Rock-Paper-Scissors Game", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Get computer's choice and calculate winner
    computer_gesture = computer_choice()
    winner = get_winner(prediction_text, computer_gesture)
    
    if winner == 'Player':
        player_score += 1
    elif winner == 'Computer':
        computer_score += 1

    # Show round result
    result_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(result_frame, f"You: {prediction_text}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(result_frame, f"Computer: {computer_gesture}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
    
    # Use if-elif-else to handle the winner message
    if winner == 'Player':
        cv2.putText(result_frame, f"Winner: You", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    elif winner == 'Computer':
        cv2.putText(result_frame, f"Winner: Computer", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    else:
        cv2.putText(result_frame, f"It's a Draw!", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
    
    cv2.putText(result_frame, f"Score - You: {player_score} | Computer: {computer_score}", (20, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Rock-Paper-Scissors Game", result_frame)
    cv2.waitKey(3000)


    # Check if anyone reached 5 points
    if player_score == 5 or computer_score == 5:
        final_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        winner_text = "You Win!" if player_score == 5 else "Computer Wins!"
        cv2.putText(final_frame, winner_text, (500, 250), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)
        cv2.putText(final_frame, "Press ESC to exit", (400, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("Rock-Paper-Scissors Game", final_frame)
        if cv2.waitKey(0) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()




