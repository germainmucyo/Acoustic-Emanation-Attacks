import numpy as np
import soundfile as sf
import math
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Function to extract key strokes
def extractKeyStroke(fileName, maxClicks, threshold):
    arr, _ = sf.read(fileName)
    arr = arr.T

    rawSound = arr[549:]  

    # Window and click parameters
    winSize = 40
    clickSize = int(44100 * 0.08)  # 0.08 seconds
    numWindows = len(rawSound) // winSize
    binSums = np.zeros(numWindows)

    for i in range(numWindows):
        currentWindow = np.fft.fft(rawSound[winSize * i:winSize * (i + 1)])
        binSums[i] = np.sum(np.abs(currentWindow))

    # click positions
    clickPositions = []
    h = 0
    while h < len(binSums):
        if binSums[h] > threshold:
            clickPositions.append((h + 1) * winSize)
            h += math.ceil(clickSize / winSize)
        else:
            h += 1

    # Extract key strokes
    keys = []
    for pos in clickPositions:
        start = int(pos - 101)
        end = start + clickSize
        if 0 <= start < len(rawSound) and end <= len(rawSound):
            keys.append(rawSound[start:end])

    # Extract the push peak which lasts 10ms
    push_peak_size = 441
    pushPeak = []
    for key in keys:
        pushPeak.append(key[:push_peak_size])

    return np.array(pushPeak), len(pushPeak), keys

# Map letters to integers
letter_to_int = {chr(97 + i): i for i in range(26)}
int_to_letter = {i: chr(97 + i) for i in range(26)}

# Thresholds for each letter
threshold_to_letter = {
    'a': 13.5, 'b': 5.5, 'c': 6.5, 'd': 13, 'e': 7, 'f': 2.5, 'g': 2.5,
    'h': 2, 'i': 3.5, 'j': 3, 'k': 4, 'l': 4, 'm': 5, 'n': 5.5, 'o': 12.5,
    'p': 14.5, 'q': 10.5, 'r': 3.5, 's': 12, 't': 3, 'u': 4, 'v': 5,
    'w': 11, 'x': 2.5, 'y': 7.5, 'z': 3.0
}

# Parameters
max_clicks = 100

# Prepare training data
print(f"Extracting Input Data...(wait)")
X, y = [], []
for char in range(26):  # For a-z
    letter = chr(97 + char)
    file_name = f"C:\\Users\\Germain\\Downloads\\lab-4-germainmucyo-main\\lab-4-germainmucyo-main\\handout\\data\\data\\{letter}.wav"
    try:
        clicks, _, _ = extractKeyStroke(file_name, max_clicks, threshold_to_letter[letter])
        for click in clicks:
            # Feature extraction
            fft_features = np.abs(np.fft.fft(click))
            fft_features = fft_features[:len(fft_features)//2]  # I used the first half
            X.append(fft_features)
            y.append(letter_to_int[letter]) 
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Train the neural network
print("____Bulding the neural network using MPL classifier____")
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X, y)

# Parameters for test data
max_clicks_test = 15

# Prepare test data
print(f"Processing test data for all letters... (wait)")
X_test, y_test = [], []
for char in range(26):  # For a-z
    letter = chr(97 + char)
    file_name = f"C:\\Users\\Germain\\Downloads\\lab-4-germainmucyo-main\\lab-4-germainmucyo-main\\handout\\data\\data\\{letter}-test.wav"
    try:
        clicks, _, _ = extractKeyStroke(file_name, max_clicks_test, threshold_to_letter[letter])
        for click in clicks:
            # Feature extraction
            fft_features = np.abs(np.fft.fft(click))
            fft_features = fft_features[:len(fft_features)//2]  # Use only the first half
            X_test.append(fft_features)
            y_test.append(letter_to_int[letter])  # Corresponding label as integer
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Convert lists to numpy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)

# Evaluate the model on test data
print("___Evaluating the model on test data___")
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Function to find secrets
def find_secrets(mlp):
    for i in range(3):  # I wanted to process on all secret files at once.
        print(f"__Processing secret{i}.wav__")
        try:
            clicks, _, _ = extractKeyStroke(f"C:\\Users\\Germain\\Downloads\\lab-4-germainmucyo-main\\lab-4-germainmucyo-main\\handout\\data\\data\\secret{i}.wav", 8, 7)
            X_secret = []
            for click in clicks:
                # Feature extraction
                fft_features = np.abs(np.fft.fft(click))
                fft_features = fft_features[:len(fft_features)//2]  # Use only the first half
                X_secret.append(fft_features)
            X_secret = np.array(X_secret)

            # Check if X_secret is 2D
            if X_secret.ndim != 2:
                n_samples = X_secret.shape[0]
                n_features = X_secret.shape[1]
                X_secret = X_secret.reshape(n_samples, n_features)

            # Predict the output probabilities
            y_secret_prob = mlp.predict_proba(X_secret)

            # Collect results for formatted output
            print(f"Predicted output for secret{i}.wav:")
            for char_index in range(len(y_secret_prob)):
                # Get the sorted indices of the top 3 predictions, Highest, Second and Third
                top_indices = np.argsort(y_secret_prob[char_index])[-3:][::-1]
                result = f"char{char_index} "
                result += " ".join([f"highest:{int_to_letter[top_indices[0]]} second largest:{int_to_letter[top_indices[1]]} third largest:{int_to_letter[top_indices[2]]}"])
                print(result)

        except Exception as e:
            print(f"Error processing secret{i}.wav: {e}")

# Call the function to find secrets
find_secrets(mlp)
