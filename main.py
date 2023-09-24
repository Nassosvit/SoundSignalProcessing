import os
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import librosa

# Ορίζουμε μια λίστα που θα περιέχει τα ονόματα-ετικέτες των φακέλων που βρίσκονται στο dataset γιά να μπορούμε στη
# συνέχεια να κάνουμε μια επανάληψη με βάση αυτή τη λίστα και να συλλέξουμε όλα τα δεδομένα
datasetDirFolder = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# Ορίζουμε το γενικό directory όπου βρίσκονται ολα τα δεδομένα μας
datasetDirectory = 'Data/SpeechCommands/speech_commands_v0.02'

# Καθορίζουμε τον αριθμό των δειγμάτων ήχου που θα πάρουμε προς επεξεργασία
samples = 2000  # Μπορούμε να αλλάξουμε το μέγεθος οσο θέλουμε

'''
Συλλέγουμε τα directory paths των αρχείων ήχου και τις αντίστοιχες ετικέτες-ονόματα τους
'''
audio_fullpaths = []
audio_digit_labels = []

for digitFolder in datasetDirFolder:
    # Κατασκευάζουμε το fullpath του καταλόγου για τον τρέχων φάκελο-νούμερο
    digit_fullpath = os.path.join(datasetDirectory, digitFolder)

    # Παίρνουμε μια λίστα με τα ονόματα των αρχείων στον κατάλογο του τρέχων φακέλου-νούμερο
    digit_file_names = os.listdir(digit_fullpath)

    # Κάνουμε μια επανάληψη στα αρχεία που βρήκαμε απο πάνω και δημιουργούμε τα fullpath τους
    files_fullpath = [os.path.join(digit_fullpath, file) for file in digit_file_names]

    # τα περνάμε στις τελικές λίστες που μας ενδιαφέρουν
    audio_fullpaths.extend(files_fullpath)
    audio_digit_labels.extend([digitFolder] * len(files_fullpath))

'''
Επιλέγουμε τυχαία ένα υποσύνολο δειγμάτων ήχου
'''
random.seed(42)  # Ορίζουμε ενα τυχαίο seed για την αναπαραγωγή αριθμών της random
random_selected_samples = random.sample(range(len(audio_fullpaths)), samples)

selected_audio_fullpaths = []
selected_audio_digit_labels = []

# Κρατάμε τα επιλεγμένα αρχεία ήχου και ετικέτες
for x in random_selected_samples:
    selected_audio_fullpaths.append(audio_fullpaths[x])
    selected_audio_digit_labels.append(audio_digit_labels[x])

print("Total audio samples:", len(selected_audio_fullpaths))

'''
Φορτώνουμε τα δεδομένα ήχου, εφαρμόζουμε το pre-emphasis φίλτρο και εξάγουμε τα mfcc χαρακτηριστικά
'''
mfcc_features = []
max_mfcc_length = 0  # Πρέπει να βρούμε το μέγιστο μήκος από τα κανονικοποιημένα mfcc χαρακτηριστικά
for i, path in enumerate(selected_audio_fullpaths):
    print(f"Processing audio {i + 1}/{len(selected_audio_fullpaths)}")

    # Φορτώνουμε τα δεδομένα ήχου καθορίζοντας την επιθυμητή συχνότητα δειγματοληψίας του ήχου σε 16000 Hz
    audio_signal, _ = librosa.load(path, sr=16000)

    # Εφαρμόζουμε το pre-emphasis
    preemphasized_audio = librosa.effects.preemphasis(audio_signal)

    #  Αφαιρούμε τη σιγή από το ηχητικό σήμα με κατώφλι 20 db
    filtered_audio, _ = librosa.effects.trim(preemphasized_audio, top_db=20)

    # Προεπεξεργασία ήχου σε χαρακτηριστικά MFCC
    mfcc = librosa.feature.mfcc(filtered_audio, sr=16000, n_mfcc=13)
    mfcc_normalized = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # κανονικοποιούμε το mfcc χαρακτηριστικό
    mfcc_features.append(mfcc_normalized)

    # Αν το τρέχων mfcc χαρακατηριστικό είναι μεγαλύτερο ενημερώνουμε την μεταβλητή που κρατάει το μέγιστο μήκος
    current_mfcc_length = mfcc_normalized.shape[1]
    if current_mfcc_length > max_mfcc_length:
        max_mfcc_length = current_mfcc_length

print("Feature extraction completed.")

'''
Προσαρμόζουμε τις ακολουθίες MFCC χαρακτηριστικών ώστε να έχουν όλα ίσο μήκος.
'''
X = []
for mfcc in mfcc_features:
    # βρισκουμε πόσο ακόμα πρέπει να σημπληρώσουμε για να είναι ίσο με το μέγιστο και το σημπληρώνουμε
    remained_length = max_mfcc_length - mfcc.shape[1]
    padded_mfcc = np.pad(mfcc, ((0, 0), (0, remained_length)), mode='constant', constant_values=0)
    X.append(padded_mfcc.T)
X = np.array(X)

# Μετατρέπουμε τα δεδομένα σε numpy πίνακες
y = np.array(selected_audio_digit_labels)

# Διαχωρίζουμε το σύνολο δεδομένων σε σύνολα εκπαίδευσης(training) και δοκιμών(testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Αναδιαμορφώνουμε τα δεδομένα εκπαίδευσης σε δυσδιάστατα
n_samples_train, _, n_features_train = X_train.shape
X_train_reshaped = X_train.reshape((n_samples_train, -1))

# Χρησιμοπούμε τον StandardScaler() για να κανονικοποιήσουμε τα δεδομένα εκπαίδευσης
standardScaler = StandardScaler()
scaled_X_train = standardScaler.fit_transform(X_train_reshaped)

# Δημιουργούμε έναν ταξινομητή SVC
svc = SVC()

# Εκπαιδεύουμε το SVC μοντέλο
svc.fit(scaled_X_train, y_train)

# Αναδιαμορφώνουμε τα δεδομένα δοκιμών σε δυσδιάστατα
n_samples_test, _, n_features_test = X_test.shape
X_test_reshaped = X_test.reshape((n_samples_test, -1))

# Χρησιμοπούμε τον StandardScaler() για να κανονικοποιήσουμε τα δεδομένα δοκιμών όπως κάναμε και με τα εκπαίδευσης
scaled_X_test = standardScaler.transform(X_test_reshaped)

# Προβλέπουμε τις ετικέτες για το σετ δοκιμών με τον ταξινομητή SVC
y_pred = svc.predict(scaled_X_test)

# Calculate the accuracy
accuracyScore = accuracy_score(y_test, y_pred)
print("Test accuracy:", accuracyScore)

'''
Τώρα φορτώνουμε το ηχογραφημένο αρχείο ήχου μας και ακολουθούμε την ιδία διαδικασία και λειτουργίες που κάναμε 
και πριν για τα αρχείο ήχου του dataset και αναγνωρίζουμε τα ψηφία απο την ηχογράφηση μας
'''

# Φορτώνουμε το ηχογραφημένο απο εμάς αρχείο .wav
recorded_audio = 'Recording.wav'
audio, _ = librosa.load(recorded_audio, sr=16000)

# Εφαρμόζουμε το pre-emphasis
audio_preemphasized = librosa.effects.preemphasis(audio)

# Αφαιρούμε τη σιγή από το ηχητικό σήμα με κατώφλι 20 db
filtered_audio, _ = librosa.effects.trim(audio_preemphasized, top_db=20)

# Χωρίζουμε το ηχογραφημένο αρχείο ήχου σε τμήματα
audio_segments = librosa.effects.split(filtered_audio, top_db=20)

# Επεξεργαζόμαστε κάθε τμήμα ήχου και αναγνωρίζουμε τα ψηφία
recognized_digits = []
for segment_start, segment_end in audio_segments:
    segment = filtered_audio[segment_start:segment_end]

    # Προεπεξεργασία τμήματος ήχου σε χαρακτηριστικά MFCC
    mfcc = librosa.feature.mfcc(segment, sr=16000, n_mfcc=13)
    mfcc_normalized = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Προσαρμόζουμε τις ακολουθίες MFCC χαρακτηριστικών ώστε να έχουν όλα ίσο μήκος.
    remained_length = max_mfcc_length - mfcc_normalized.shape[1]
    padded_mfcc = np.pad(mfcc_normalized, ((0, 0), (0, remained_length)), mode='constant', constant_values=0)

    # Αναδιαμορφώνουμε τα προσαρμοσμένα δεδομένα και χρησιμοποιούμε τον StandardScaler() για να τα κανονικοποιήσουμε
    reshaped_mfcc = padded_mfcc.T.reshape(1, -1)
    scaled_mfcc = standardScaler.transform(reshaped_mfcc)

    # Προβλέπουμε την ετικέτα-ψηφίο για το τρέχων τμήμα ήχου
    predicted_digit = svc.predict(scaled_mfcc)
    recognized_digits.append(predicted_digit[0])

print("Recognized digits:", recognized_digits)
