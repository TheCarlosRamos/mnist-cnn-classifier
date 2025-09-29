#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import sys
import matplotlib.pyplot as plt

def load_data():
    try:
        with open('mnist_784.csv', 'r') as fid:
            fid.readline()
            data = np.loadtxt(fid, delimiter=',')
            x = data[:, :-1].reshape(-1, 28, 28, 1) / 255.0
            y = data[:, -1].astype(int)
            return x, y, x, y
    except FileNotFoundError:
        print("\nErro: Arquivo 'mnist_784.csv' não encontrado.")
        sys.exit(1)

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    x_train, y_train, x_test, y_test = load_data()
    
    model_path = 'mnist_cnn.h5'
    if os.path.exists(model_path):
        print("\nCarregando modelo existente...")
        model = models.load_model(model_path)
    else:
        print("\nTreinando novo modelo...")
        model = build_model()
        model.fit(x_train, y_train,
                 epochs=10,
                 batch_size=128,
                 validation_split=0.2,
                 verbose=1)
        model.save(model_path)
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nAcurácia no Teste: {test_acc:.4f}")
    print(f"Loss no Teste: {test_loss:.4f}")
    
    # Predições
    y_pred = np.argmax(model.predict(x_test), axis=1)
    
    # ===============================
    # Cálculo manual da matriz de confusão
    # ===============================
    num_classes = max(y_test.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(y_test)):
        true_class = y_test[i]
        pred_class = y_pred[i]
        cm[true_class][pred_class] += 1

    # Impressão da matriz
    print("\nMatriz de Confusão: \n")
    print("   " + " ".join(f"{i:^5}" for i in range(num_classes)))
    for i, row in enumerate(cm):
        print(f"{i} | " + " ".join(f"{count:5}" for count in row))
    
    # ===============================
    # Fazer predição com a imagem
    # ===============================
    sample_idx = np.random.randint(len(x_test))
    sample_pred = model.predict(x_test[sample_idx][np.newaxis,...])[0]
    predicted_class = np.argmax(sample_pred)
    
    print("\nExemplo de Predição:")
    print(f"Índice: {sample_idx} | Real: {y_test[sample_idx]} | Previsto: {predicted_class}")
    print("Probabilidades:")
    for i, prob in enumerate(sample_pred):
        print(f"  {i}: {prob*100:6.2f}% {'←' if i == y_test[sample_idx] else ''}")

    # Plotar a imagem correspondente
    plt.imshow(x_test[sample_idx].reshape(28, 28), cmap="gray")
    plt.title(f"Real: {y_test[sample_idx]} | Previsto: {predicted_class}")
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    main()



