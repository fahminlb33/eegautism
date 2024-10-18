# Model Architecture Archive

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=2, padding="same"),
    tf.keras.layers.Activation("swish"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=3),

    tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=2, padding="same"),
    tf.keras.layers.Activation("swish"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
```

```python
# conv - hi-freq
x1 = tf.keras.layers.Conv1D(32, kernel_size=2, strides=2, activation="relu", padding="same")(inputs)
x2 = tf.keras.layers.Conv1D(32, kernel_size=4, strides=2, activation="relu", padding="same")(inputs)
x3 = tf.keras.layers.Conv1D(32, kernel_size=8, strides=2, activation="relu", padding="same")(inputs)

c1 = tf.keras.layers.Concatenate(axis=2)([x1, x2, x3])

x1 = tf.keras.layers.Conv1D(32, kernel_size=2, strides=2, activation="relu", padding="same")(c1)
x2 = tf.keras.layers.Conv1D(32, kernel_size=4, strides=2, activation="relu", padding="same")(c1)
x3 = tf.keras.layers.Conv1D(32, kernel_size=8, strides=2, activation="relu", padding="same")(c1)

c2 = tf.keras.layers.Concatenate(axis=2)([x1, x2, x3])

x1 = tf.keras.layers.Conv1D(32, kernel_size=2, strides=2, activation="relu", padding="same")(c2)
x2 = tf.keras.layers.Conv1D(32, kernel_size=4, strides=2, activation="relu", padding="same")(c2)
x3 = tf.keras.layers.Conv1D(32, kernel_size=8, strides=2, activation="relu", padding="same")(c2)

c3 = tf.keras.layers.Concatenate(axis=2)([x1, x2, x3])

# gru
g1 = tf.keras.layers.GRU(32, return_sequences=True)(c3)
g2 = tf.keras.layers.GRU(32, return_sequences=True)(g1)

gc1 = tf.keras.layers.Concatenate(axis=2)([g1, g2])

g3 = tf.keras.layers.GRU(32, return_sequences=True)(gc1)

gc2 = tf.keras.layers.Concatenate(axis=2)([g1, g2, g3])

g4 = tf.keras.layers.GRU(32)(gc2)
```

```python
inputs = tf.keras.layers.Input(shape=(15361, 16,))

cA1, cD1 = DWT(name="decomposition_1")(inputs)

# cD1
# x1 = ConvolutionalBlock()(cD1)
# x1 = ConvolutionalBlock(pool_size=2, strides=2)(x1)
# x1 = ConvolutionalBlock(pool_size=2, strides=2)(x1)
# x1 = ConvolutionalBlock(pool_size=2, strides=2)(x1)
x1 = tf.keras.layers.Conv1D(32, kernel_size=8, padding="same")(cD1)
x1 = tf.keras.layers.Conv1D(32, kernel_size=8, padding="same")(x1)
x1 = tf.keras.layers.Conv1D(32, kernel_size=8, padding="same")(x1)
x1 = tf.keras.layers.MaxPool1D(pool_size=4)(x1)
# x1 = ConvolutionalBlock()(x1)
# x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)
x1 = tf.keras.layers.Flatten()(x1)
x1 = tf.keras.layers.Dense(512, activation="relu")(x1)

cA2, cD2 = DWT(name="decomposition_2")(cA1)

# cD2
x2 = tf.keras.layers.Conv1D(32, kernel_size=8, padding="same")(cD2)
x2 = tf.keras.layers.Conv1D(32, kernel_size=8, padding="same")(x2)
x2 = tf.keras.layers.Conv1D(32, kernel_size=8, padding="same")(x2)
x2 = tf.keras.layers.MaxPool1D(pool_size=4)(x2)
# x2 = ConvolutionalBlock()(x2)
# x2 = tf.keras.layers.GlobalAveragePooling1D()(x2)
x2 = tf.keras.layers.Flatten()(x2)
x2 = tf.keras.layers.Dense(1024, activation="relu")(x2)
x2 = tf.keras.layers.Dense(512, activation="relu")(x2)

cA3, cD3 = DWT(name="decomposition_3")(cA2)

# cD3
x3 = tf.keras.layers.Conv1D(32, kernel_size=8, padding="same")(cD3)
x3 = tf.keras.layers.Conv1D(32, kernel_size=8, padding="same")(x3)
x3 = tf.keras.layers.Conv1D(32, kernel_size=8, padding="same")(x3)
x3 = tf.keras.layers.MaxPool1D(pool_size=4)(x3)
# x3 = tf.keras.layers.GlobalAveragePooling1D()(x3)
x3 = tf.keras.layers.Flatten()(x3)
x3 = tf.keras.layers.Dense(1024, activation="relu")(x3)
x3 = tf.keras.layers.Dense(512, activation="relu")(x3)

# cA3
x4 = tf.keras.layers.Conv1D(32, kernel_size=8, padding="same")(cA3)
x4 = tf.keras.layers.Conv1D(32, kernel_size=8, padding="same")(x4)
x4 = tf.keras.layers.Conv1D(32, kernel_size=8, padding="same")(x4)
x4 = tf.keras.layers.MaxPool1D(pool_size=4)(x4)
# x4 = tf.keras.layers.GlobalAveragePooling1D()(x4)
x4 = tf.keras.layers.Flatten()(x4)
x4 = tf.keras.layers.Dense(1024, activation="relu")(x4)
x4 = tf.keras.layers.Dense(512, activation="relu")(x4)

conc = tf.keras.layers.concatenate([
    x1,
    x2,
    x3,
    x4
])

out = tf.keras.layers.Flatten()(conc)
out = tf.keras.layers.Dense(1024, activation="relu")(out)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(out)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```
