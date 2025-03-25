import cv2
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage

# Function to load images from the data folder and prepare them for recognition
def load_images(data_folder):
    faces = []
    labels = []
    label_names = []

    for label, name in enumerate(os.listdir(data_folder)):
        image_path = os.path.join(data_folder, name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            faces.append(image)
            labels.append(name.split(".")[0])  # Extracting name without extension
            label_names.append(name)

    return faces, labels

# Function for face recognition
def recognize_face(frame, faces_detected, known_faces, known_labels):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in faces_detected:
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Recognize the face using known faces
        label = "Unknown"
        for i, face in enumerate(known_faces):
            result = cv2.matchTemplate(roi_gray, face, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, _ = cv2.minMaxLoc(result)
            if confidence > 0.7:  # Set confidence threshold
                label = known_labels[i]
                break

        # Draw rectangle and label around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

# Function to send email with attached image using SMTP with SSL
def send_email_alert(sender_email, receiver_email, subject, body, email_password, image_path=None):
    try:
        em = EmailMessage()
        em['From'] = sender_email
        em['To'] = receiver_email
        em['Subject'] = subject
        em.set_content(body)

        if image_path is not None:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                em.add_attachment(image_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(sender_email, email_password)
            smtp.send_message(em)

        print("Email alert sent successfully!")
    except Exception as e:
        print(f"Error sending email alert: {e}")

        # Exit the program
        sys.exit()


# Main function
def main():
    data_folder = "C:\\Users\\bhagya\\Desktop\\finalyearproject1\\finalproject\\data"  # change the path as your local path
    intruder_folder = "intruder"  # Folder to save images of intruders
    sender_email = "senderem000@gmail.com"  # Sender's email address
    receiver_email = "receiver876@gmail.com"  # Recipient's email address
    email_password = "rdpbkeierskgrcqv"  # Your email password

    # Create intruder folder if it doesn't exist
    if not os.path.exists(intruder_folder):
        os.makedirs(intruder_folder)

    # Load known faces from data folder
    known_faces, known_labels = load_images(data_folder)

    # Initialize the Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Open the camera
    cap = cv2.VideoCapture(0)

    # Counter to keep track of intruder images captured
    intruder_count = 0

    while True:
        ret, frame = cap.read()  # Capture a frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over detected faces
        for (x, y, w, h) in faces:
            if intruder_count < 2:
                # Save the unknown face image in the intruder folder
                image_path = os.path.join(intruder_folder, f"intruder_{intruder_count + 1}.jpg")
                cv2.imwrite(image_path, frame)

                # Send email alert with the intruder image attached
                subject = "Intruder Detected!"
                body = "Motion detected! Intruder detected in the premises. Please check the security camera"
                # Send email alert with the intruder image attached
                image_path = os.path.join(intruder_folder, f"intruder_{intruder_count + 1}.jpg")
                send_email_alert(sender_email, receiver_email, subject, body, email_password, image_path)

                intruder_count += 1
                print("Warning Sent")

                # Close the window after sending the email
                cv2.destroyAllWindows()
                sys.exit()  # Exit the program
            else:
                print("Maximum number of intruder images captured.")
                break

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Check for intruder (Unknown face)
        if "Unknown" in known_labels:
            unknown_index = known_labels.index("Unknown")
            if "Unknown" in [label for _, label, _, _ in faces]:
                # Send email alert with the intruder image attached
                subject = "Intruder Detected!"
                body = "Motion detected! Intruder detected in the premises. Please check the security camera."
                send_email_alert(sender_email, receiver_email, subject, body, email_password)

        # Wait for user to press a key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
