import cv2

# Function to run the camera feed
def run_camera():
    st.title("Real-Time Anomaly Detection")
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Convert the frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st.image(image, caption='Live Feed', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction, axis=1)

        # Display result
        if class_index == 0:
            st.write("This product is Normal.")
        else:
            st.write("Anomaly detected in this product.")

    camera.release()

# Add a button to start the camera feed
if st.button("Start Camera"):
    run_camera()
