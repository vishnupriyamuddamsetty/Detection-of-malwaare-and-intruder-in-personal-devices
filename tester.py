import cv2
import os
import smtplib
import ssl
from email.message import EmailMessage

# Function to load images from the data folder and prepare them for recognition
def load_images(data_folder):
    faces = []
    labels = []

    for name in os.listdir(data_folder):
        image_path = os.path.join(data_folder, name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            faces.append(image)
            labels.append(name.split(".")[0])  # Extracting name without extension

    return faces, labels

# Function to send email using SMTP with SSL
def send_email_alert(sender_email, receiver_email, subject, body, email_password, image_paths):
    try:
        em = EmailMessage()
        em['From'] = sender_email
        em['To'] = receiver_email
        em['Subject'] = subject
        em.set_content(body)

        # Attach each intruder image to the email
        for image_path in image_paths:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                img_name = os.path.basename(image_path)
                em.add_attachment(img_data, maintype='image', subtype='jpg', filename=img_name)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(sender_email, email_password)
            smtp.send_message(em)

        print("Email alert sent successfully!")
    except Exception as e:
        print(f"Error sending email alert: {e}")


# Main function
def main():
    data_folder = "C:/Users/mudda/OneDrive/Desktop/finalproject/data"  # change the path as your local path
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
            if intruder_count < 5:
                # Save the unknown face image in the intruder folder
                image_path = os.path.join(intruder_folder, f"intruder_{intruder_count + 1}.jpg")
                cv2.imwrite(image_path, frame)

                intruder_count += 1

        # Check for intruder (Unknown face)
        if "Unknown" in known_labels:
            unknown_index = known_labels.index("Unknown")
            if "Unknown" in [label for _, label, _, _ in faces]:
                # Send email alert with the intruder images attached
                subject = "Intruder Detected!"
                body = "Motion detected! Intruder detected in the premises. Please check the security camera."
                send_email_alert(sender_email, receiver_email, subject, body, email_password, [os.path.join(intruder_folder, f"intruder_{i}.jpg") for i in range(1, intruder_count + 1)])

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Wait for user to press a key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
