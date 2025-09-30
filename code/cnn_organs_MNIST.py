#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import sys
import matplotlib.pyplot as plt

import medmnist
from medmnist import INFO
from torchvision import transforms

DATASET_NAME = "organamnist"

CLASS_NAMES = {
    "organamnist": [
        "Fígado", "Rim", "Baço", "Pâncreas", "Aorta", 
        "Vesícula", "Pulmão", "Adrenal", "Bexiga", "Intestino/Osso"
    ],
    "organcmnist": [
        "Fígado", "Rim", "Baço", "Pâncreas", "Aorta", 
        "Vesícula", "Pulmão", "Adrenal", "Bexiga", "Intestino/Osso"
    ],
    "organsmnist": [
        "Fígado", "Rim", "Baço", "Pâncreas", "Aorta", 
        "Vesícula", "Pulmão", "Adrenal", "Bexiga", "Intestino/Osso"
    ],
    "bloodmnist": [
        "Basófilo", "Eosinófilo", "Eritroblasto", "Granulócito Imaturo",
        "Linfócito", "Monócito", "Neutrófilo", "Plaqueta"
    ]
}

def get_class_names(dataset_name, n_classes):
    """Retorna os nomes das classes para o dataset"""
    if dataset_name in CLASS_NAMES:
        names = CLASS_NAMES[dataset_name]
        if len(names) == n_classes:
            return names
        else:
            print(f"Aviso: Mapeamento tem {len(names)} classes, mas dataset tem {n_classes}")
            if len(names) < n_classes:
                names += [f"Classe {i}" for i in range(len(names), n_classes)]
            else:
                names = names[:n_classes]
            return names
    else:
        return [f"Classe {i}" for i in range(n_classes)]

def load_data(dataset_name=DATASET_NAME):
    if dataset_name not in INFO:
        available_datasets = list(INFO.keys())
        raise ValueError(f"Dataset '{dataset_name}' não encontrado. Chaves disponíveis: {available_datasets}")
    
    info = INFO[dataset_name]
    print(f"Estrutura do INFO: {info}")  
    
    DataClass = getattr(medmnist, info["python_class"])
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    train_dataset = DataClass(split="train", transform=data_transform, download=True)
    test_dataset = DataClass(split="test", transform=data_transform, download=True)
    
    x_train = np.array([d[0].numpy().transpose(1,2,0) for d in train_dataset])
    y_train = np.array([d[1] for d in train_dataset])
    x_test = np.array([d[0].numpy().transpose(1,2,0) for d in test_dataset])
    y_test = np.array([d[1] for d in test_dataset])
    
    if len(y_train.shape) > 1 and y_train.shape[1] == 1:
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    
    x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())
    
    n_classes = len(np.unique(y_train))
    n_channels = x_train.shape[-1]
    
    return x_train, y_train, x_test, y_test, n_channels, n_classes, info

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    x_train, y_train, x_test, y_test, n_channels, n_classes, info = load_data(DATASET_NAME)
    input_shape = (x_train.shape[1], x_train.shape[2], n_channels)
    
    class_names = get_class_names(DATASET_NAME, n_classes)
    
    print(f"\nDataset: {DATASET_NAME}")
    print(f"Classes: {n_classes} | 🖼️ Canais: {n_channels} | 🔢 Imagens treino: {len(x_train)} | teste: {len(x_test)}")
    print(f"Forma dos dados de treino: {x_train.shape}")
    print(f"Forma dos labels de treino: {y_train.shape}")
    
    print(f"\nNomes das Classes:")
    for i, name in enumerate(class_names):
        print(f"   {i}: {name}")
    
    model_path = f'{DATASET_NAME}_cnn.h5'
    if os.path.exists(model_path):
        print("\nCarregando modelo existente...")
        model = models.load_model(model_path)
    else:
        print("\nTreinando novo modelo...")
        model = build_model(input_shape, n_classes)
        model.summary()
        history = model.fit(x_train, y_train,
                          epochs=10,
                          batch_size=128,
                          validation_split=0.2,
                          verbose=1)
        model.save(model_path)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nAcurácia no Teste: {test_acc:.4f}")
    print(f"Loss no Teste: {test_loss:.4f}")
    
    y_pred = np.argmax(model.predict(x_test), axis=1)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_test)):
        if y_test[i] < n_classes and y_pred[i] < n_classes: 
            cm[y_test[i], y_pred[i]] += 1
    
    print("\nMatriz de Confusão\n")
    print("   " + " ".join(f"{i:^5}" for i in range(n_classes)))
    for i, row in enumerate(cm):
        print(f"{i} | " + " ".join(f"{count:5}" for count in row))
    
    sample_idx = np.random.randint(len(x_test))
    sample_pred = model.predict(x_test[sample_idx][np.newaxis,...])[0]
    predicted_class = np.argmax(sample_pred)
    
    real_class_idx = y_test[sample_idx]
    if real_class_idx >= len(class_names):
        real_class_name = f"Classe {real_class_idx}"
    else:
        real_class_name = class_names[real_class_idx]
    
    if predicted_class >= len(class_names):
        predicted_class_name = f"Classe {predicted_class}"
    else:
        predicted_class_name = class_names[predicted_class]
    
    print(f"\nExemplo de Predição:")
    print(f"Índice: {sample_idx}")
    print(f"Real: {real_class_idx} - {real_class_name}")
    print(f"Previsto: {predicted_class} - {predicted_class_name}")
    print(f"Confiança: {sample_pred[predicted_class]:.2%}")
    
    top3_indices = np.argsort(sample_pred)[-3:][::-1]
    print(f"\nTop 3 Predições:")
    for i, idx in enumerate(top3_indices):
        if idx < len(class_names):
            class_name = class_names[idx]
        else:
            class_name = f"Classe {idx}"
        print(f"  {i+1}º: {class_name} - {sample_pred[idx]:.2%}")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(x_test[sample_idx].squeeze(), cmap="gray" if n_channels==1 else None)
    plt.title(f"Real: {real_class_name}\nPrevisto: {predicted_class_name}\nConfiança: {sample_pred[predicted_class]:.2%}")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    colors = ['red' if i == predicted_class else 'blue' for i in range(len(sample_pred))]
    bars = plt.bar(range(len(sample_pred)), sample_pred, color=colors)
    plt.title('Probabilidades por Classe')
    plt.xlabel('Classe')
    plt.ylabel('Probabilidade')
    plt.xticks(range(len(sample_pred)), [f'{i}' for i in range(len(sample_pred))], rotation=45)
    
    for bar, prob in zip(bars, sample_pred):
        height = bar.get_height()
        if height > 0.01:  
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📋 Legenda Completa - {DATASET_NAME}:")
    for i, name in enumerate(class_names):
        print(f"   {i}: {name}")

if __name__ == '__main__':
    main()