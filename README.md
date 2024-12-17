# **Lab 4: Acoustic Emanation Attacks**

---

## **Overview**

This lab explores **Acoustic Emanation Attacks**, where keystroke sounds from a keyboard are analyzed to predict and recover sensitive information such as a passphrase. By leveraging audio recordings and machine learning, we implemented a system to detect and predict keystrokes, achieving notable accuracy.

---

## **1. Objective**

- Analyze keystroke sounds recorded from a keyboard.  
- Implement a system to **detect** and **predict keystrokes** based on acoustic data.  
- Recover an **8-character secret passphrase** (e.g., "password") from the provided audio file.  

---

## **2. Tools and Technologies Used**

- **Programming Language**: Python  
- **Libraries**:  
  - **Librosa**: Audio analysis and feature extraction  
  - **NumPy/Pandas**: Data processing  
  - **Scikit-Learn**: Machine learning model implementation  
  - **Matplotlib**: Visualization  
- **Audio Input**: Provided audio file containing keystroke recordings.  

---

## **3. System Accuracy**

The implemented model achieved an accuracy of:  0.6594 (65.94%)

This demonstrates the effectiveness of the acoustic analysis system in detecting and predicting keystrokes.

---

## **4. Results**

- Successfully **recovered the secret passphrase** from the audio recording:  **"password"**  

- The system effectively identified and predicted keystrokes by analyzing the acoustic emanations.

---

## **5. Challenges Faced**

1. **Audio Noise**:  
   - Background noise in the audio file sometimes interfered with keystroke detection.  

2. **Model Accuracy**:  
   - Improving the model’s performance to handle subtle variations in keystroke sounds.  

3. **Feature Extraction**:  
   - Extracting meaningful audio features required fine-tuning and experimentation with tools like **Librosa**.  

---

## **6. Lessons Learned**

- **Acoustic Emanation Vulnerabilities**:  
   - Learned how keyboards emit unique sounds for each keystroke, which can be exploited for side-channel attacks.  

- **Audio Processing**:  
   - Gained hands-on experience in audio analysis using tools like **Librosa**.  

- **Machine Learning Application**:  
   - Successfully applied machine learning techniques to predict sensitive information from audio recordings.  

---

## **7. Conclusion**

This lab demonstrated the feasibility of **Acoustic Emanation Attacks** for recovering sensitive data, such as passphrases, from keystroke sounds. By analyzing the audio recording, we achieved:  

- **65.94% accuracy** in keystroke prediction.  
- Successful recovery of the passphrase: **"password"**.  

This highlights the potential risks of acoustic side-channel attacks and the importance of securing sensitive environments against such vulnerabilities.

---

## **Author**  
**Germain Mucyo**  
**Email**: [mucyo.g@northeastern.edu](mailto:mucyo.g@northeastern.edu)  


