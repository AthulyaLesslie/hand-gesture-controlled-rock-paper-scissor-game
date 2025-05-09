import streamlit as st
import subprocess

st.set_page_config(page_title="Rock Paper Scissors Game", layout="centered")

# Title and Instructions
st.title("🪨 📄 ✂️ Rock Paper Scissors Game")

st.markdown("""
Welcome to the **Hand Gesture Controlled Rock-Paper-Scissors Game**!  
Use your webcam to show your hand gesture:  
- ✊ Rock  
- 🖐️ Paper  
- ✌️ Scissors  

Click the button below to start the game. The webcam will open and detect your gesture in real-time.
""")

# Button to start the game
if st.button('🎮 Start Game'):
    with st.spinner('Opening webcam and running the game...'):
        subprocess.run(["python", "GAME.py"])
        st.success('✅ Game Over! You can play again by clicking the button.')





