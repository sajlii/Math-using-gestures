import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
import handTrackingModule as ht
import streamlit as st

st.set_page_config(page_title="Math with Gestures using AI",layout="wide")
st.title("Math with gestures using AI")
col1,col2=st.columns([3,2])

with col1:
    run=st.checkbox('Run',value =True)
    FRAME_WINDOW=st.image([])
with col2:
    st.title("Response from AI")
    output_text_area=st.subheader("")
genai.configure(api_key="AIzaSyAe7XhteaObUsoc0bjOTA94KQMhw_omGJw")
model = genai.GenerativeModel('gemini-1.5-flash')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = ht.handDetector()
canvas = None
prev_pos = None
response=""

def getHandInfo(frame):
    frame = detector.findHands(frame)
    hands, frame = detector.findPosition(frame, draw=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas, frame):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][1:3]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(frame)
    return current_pos, canvas

def sendtoAI(model, canvas, fingers):
    if fingers == [0, 1, 1, 1, 1]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this problem", pil_image])
        return response.text

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret:
        info = getHandInfo(frame)
        if canvas is None:
            canvas = np.zeros_like(frame)
        if info:
            fingers, lmList = info
            prev_pos, canvas = draw(info, prev_pos, canvas, frame)
            response = sendtoAI(model, canvas, fingers)
            if response:
                print("The response from AI model ",response)
        frame_combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        FRAME_WINDOW.image(frame_combined,channels="BGR")
        if response:
            output_text_area.text(response)
        #cv2.imshow("Live Webcam", frame_combined)
        #cv2.imshow("canvas", canvas)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()