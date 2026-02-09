import numpy as np
import pandas as pd
import tensorflow as tf
import os

SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 两个人
PERSONS = [
    "bdx", 
    "yxh",  
]
NUM_PERSONS = len(PERSONS)

ONE_HOT = np.eye(NUM_PERSONS)
MAX_FSR = 3986.0
WINDOW_SIZE = 60
STEP = 20

def make_windows_from_csv(csv_path, label_index):
    df = pd.read_csv(csv_path)
    expected_cols = [f"FSR{i}" for i in range(1, 11)]
    assert all(col in df.columns for col in expected_cols), f"loss col{expected_cols}"

    data = df[expected_cols].values.astype(np.float32)  # (T,10)
    num_rows = data.shape[0]

    windows, labels = [], []
    for start in range(0, num_rows - WINDOW_SIZE + 1, STEP):
        seg = data[start:start+WINDOW_SIZE]

        # 简单归一化（后续建议改成训练集均值方差标准化）
        seg = seg / MAX_FSR

        windows.append(seg)  # 保持 (WINDOW_SIZE,10)
        labels.append(ONE_HOT[label_index])

    return np.array(windows, np.float32), np.array(labels, np.float32)



all_inputs = []
all_outputs = []

base_dir = "./raw_data"  # 修改为你的实际路径

for idx, person in enumerate(PERSONS):
    csv_path = os.path.join(base_dir, f"{person}.csv")
    X, y = make_windows_from_csv(csv_path, label_index=idx)
    print(f"{person}: 生成 {X.shape[0]} 个窗口样本，输入维度 {X.shape[1]}")
    all_inputs.append(X)
    all_outputs.append(y)

inputs = np.concatenate(all_inputs, axis=0)
outputs = np.concatenate(all_outputs, axis=0)

print("总样本数:", inputs.shape[0])
print("输入每样本维度:", inputs.shape[1])
print("标签维度:", outputs.shape[1])


num_inputs = len(inputs)
indices = np.arange(num_inputs)
np.random.shuffle(indices)

inputs = inputs[indices]
outputs = outputs[indices]

TRAIN_SPLIT = int(0.7 * num_inputs)
TEST_SPLIT  = int(0.2 * num_inputs + TRAIN_SPLIT)

inputs_train, inputs_test, inputs_val = np.split(inputs, [TRAIN_SPLIT, TEST_SPLIT])
outputs_train, outputs_test, outputs_val = np.split(outputs, [TRAIN_SPLIT, TEST_SPLIT])

print("训练集:", inputs_train.shape[0])
print("测试集:", inputs_test.shape[0])
print("验证集:", inputs_val.shape[0])

# 构建模型：Conv1D 期望 (time, channels) => (WINDOW_SIZE, 10)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW_SIZE, 10)),
    tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same'),
    tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_PERSONS, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    inputs_train, outputs_train,
    epochs=40,
    batch_size=8,
    validation_data=(inputs_val, outputs_val)
)

test_loss, test_acc = model.evaluate(inputs_test, outputs_test)
print("测试集准确率:", test_acc)