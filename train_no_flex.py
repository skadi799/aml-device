import os
import re
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)

PERSONS = ["personx", "personw", "personh"]
NUM_PERSONS = len(PERSONS)
ONE_HOT = np.eye(NUM_PERSONS, dtype=np.float32)

base_dir = "./raw_data" 

WINDOW_SIZE = 60
STEP = 40

EPOCHS = 40
BATCH_SIZE = 8
LR = 1e-4

# CSV: 21 columns (no header)
COL_NAMES = (
    ["time"] +
    [f"FSR{i}" for i in range(1, 11)] +
    ["flex"] +
    [f"IMU{i}" for i in range(1, 10)]
)
assert len(COL_NAMES) == 21
FEATURE_COLS = [c for c in COL_NAMES if c != "time"] 

#  File split (fixed rule)

def split_data(person_dir: str):
    all_csv = sorted(glob.glob(os.path.join(person_dir, "*.csv")))
    if len(all_csv) == 0:
        raise FileNotFoundError(f"Folder empty or not found: {person_dir}")

    train_files, test_files = [], []
    for fp in all_csv:
        name = os.path.basename(fp).lower().strip()

        # Train: 1.csv ~ 10.csv
        if re.fullmatch(r"(10|[1-9])\.csv", name):
            train_files.append(fp)

        # Test: long1.csv ~ long3.csv
        elif re.fullmatch(r"long[1-3]\.csv", name):
            test_files.append(fp)


    # sort numerically for 1..10
    def key_num(p):
        m = re.match(r"(\d+)\.csv$", os.path.basename(p).lower())
        return int(m.group(1)) if m else 999

    train_files = sorted(train_files, key=key_num)
    test_files = sorted(test_files)

    return train_files, test_files


# =========================
# 4) Windowing
# =========================
def sliding_window(csv_path: str, label_index: int):

    df = pd.read_csv(
        csv_path,
        header=None,
        sep=",",
        engine="python",
        skip_blank_lines=True,
        on_bad_lines="skip"
    )

    df.columns = COL_NAMES

    data = df[FEATURE_COLS].values.astype(np.float32)
    data = np.delete(data, 10, axis=1)   # remove flex sensor
    num_rows = data.shape[0]

    windows, labels = [], []
    for start in range(0, num_rows - WINDOW_SIZE + 1, STEP):
        seg = data[start:start + WINDOW_SIZE]
        windows.append(seg)
        labels.append(ONE_HOT[label_index])

    return np.array(windows, np.float32), np.array(labels, np.float32)

def collect_dataset(person_idx: int, files):
    X_all, y_all = [], []
    for fp in files:
        X, y = sliding_window(fp, label_index=person_idx)
        if X.size > 0:
            X_all.append(X)
            y_all.append(y)
    if len(X_all) == 0:
        return np.zeros((0, WINDOW_SIZE, 20), np.float32), np.zeros((0, NUM_PERSONS), np.float32)
    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)

#  Standardization, z-score for train data
def standardize_traindata(train, test):
    flat_train = train.reshape(-1, train.shape[-1])  # (N*T,20)
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0) + 1e-8

    print("\n Standardization Parameters")
    for i, (m, s) in enumerate(zip(mean, std)):
        if i < 10: # remove flex
            feat_name = FEATURE_COLS[i]
        else:
            feat_name = FEATURE_COLS[i+1]
        print(f"{i:02d} {feat_name:>5s}: mean={m:.6f}, std={s:.6f}")
    print("\n")

    train_z = (train - mean) / std
    test_z  = (test  - mean) / std
    return train_z, test_z, mean, std

#  Build datasets
train_X_list, train_y_list = [], []
test_X_list,  test_y_list  = [], []

for idx, person in enumerate(PERSONS):
    person_dir = os.path.join(base_dir, person)
    tr_files, te_files = split_data(person_dir)


    Xtr, ytr = collect_dataset(idx, tr_files)
    Xte, yte = collect_dataset(idx, te_files)

    print(f"[{person}] windows: train={Xtr.shape[0]}, test={Xte.shape[0]}")

    train_X_list.append(Xtr); train_y_list.append(ytr)
    test_X_list.append(Xte);  test_y_list.append(yte)

inputs_train = np.concatenate(train_X_list, axis=0)
outputs_train = np.concatenate(train_y_list, axis=0)
inputs_test  = np.concatenate(test_X_list, axis=0)
outputs_test = np.concatenate(test_y_list, axis=0)

print("\n Dataset Summary ")
print("Train:", inputs_train.shape, outputs_train.shape)
print("Test :", inputs_test.shape, outputs_test.shape)

# Shuffle train
perm = np.random.permutation(len(inputs_train))
inputs_train = inputs_train[perm]
outputs_train = outputs_train[perm]

# Standardize using train data only
inputs_train, inputs_test, mean, std = standardize_traindata(inputs_train, inputs_test)

# CNN model
model_cnn = tf.keras.Sequential([
    # tf.keras.layers.Input(shape=(WINDOW_SIZE, 20)), #remove flex sensor
    tf.keras.layers.Input(shape=(WINDOW_SIZE, 19)),
    tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same'),
    tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_PERSONS, activation='softmax')
])

model_cnn.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model_cnn.fit(
    inputs_train, outputs_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=0
)

test_loss, test_acc = model_cnn.evaluate(inputs_test, outputs_test, verbose=0)
print(f'CNN Acc: {(100*test_acc):.2f} %')

#flatten temporal data
Xtr = inputs_train.reshape(inputs_train.shape[0], -1)
Xte = inputs_test.reshape(inputs_test.shape[0], -1)

ytr = np.argmax(outputs_train, axis=1)
yte = np.argmax(outputs_test, axis=1)

#Logistic Regression
model_lr = LogisticRegression(
    C=1.0,
    max_iter=2000,
    solver="lbfgs"
)

model_lr.fit(Xtr, ytr)

acc_lr = model_lr.score(Xte, yte)
print(f'LR Acc: {(100*acc_lr):.2f} %')

# SVM model
model_svm = LinearSVC(
    C=1.0,
    max_iter=5000
)

model_svm.fit(Xtr, ytr)

acc_svm = model_svm.score(Xte, yte)
print(f'Linear SVM Acc: {(100*acc_svm):.2f} %')
