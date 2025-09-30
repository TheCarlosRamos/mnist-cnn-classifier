#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

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

def predict_external_image(model, image_path, class_names, input_shape, n_channels=1):
    """
    Faz predição em uma imagem externa do computador
    
    Args:
        model: Modelo treinado
        image_path: Caminho para a imagem
        class_names: Lista com nomes das classes
        input_shape: Forma esperada pela rede (altura, largura, canais)
        n_channels: Número de canais (1 para grayscale, 3 para color)
    """
    try:
        print(f"\nProcessando imagem: {image_path}")
        
        
        with Image.open(image_path) as img:
            
            if n_channels == 1:
                img = img.convert('L')
            else:
                img = img.convert('RGB')
            
            print(f"   Dimensões originais: {img.size}")
            print(f"   Modo: {img.mode}")
            
            
            target_size = (input_shape[1], input_shape[0])
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            
            
            img_array = np.array(img_resized)
            
            
            img_normalized = img_array / 255.0
            
            
            if len(img_normalized.shape) == 2:
                img_processed = img_normalized.reshape(1, target_size[1], target_size[0], 1)
            else:
                img_processed = img_normalized.reshape(1, target_size[1], target_size[0], n_channels)
            
            print(f"   Forma para o modelo: {img_processed.shape}")
            
            
            prediction = model.predict(img_processed, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            
            if predicted_class < len(class_names):
                predicted_class_name = class_names[predicted_class]
            else:
                predicted_class_name = f"Classe {predicted_class}"
            
            
            print(f"\nRESULTADO DA PREDIÇÃO:")
            print(f"   Arquivo: {os.path.basename(image_path)}")
            print(f"   Classe Prevista: {predicted_class} - {predicted_class_name}")
            print(f"   Confiança: {confidence:.2%}")
            
            
            top3_indices = np.argsort(prediction)[-3:][::-1]
            print(f"\nTOP 3 PREDIÇÕES:")
            for i, idx in enumerate(top3_indices):
                if idx < len(class_names):
                    class_name = class_names[idx]
                else:
                    class_name = f"Classe {idx}"
                print(f"   {i+1}º: {class_name} - {prediction[idx]:.2%}")
            
            
            plt.figure(figsize=(15, 5))
            
            
            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap='gray' if n_channels == 1 else None)
            plt.title(f'Imagem Original\n{img.size}')
            plt.axis('off')
            
            
            plt.subplot(1, 3, 2)
            plt.imshow(img_resized, cmap='gray' if n_channels == 1 else None)
            plt.title(f'Redimensionada\n{img_resized.size}')
            plt.axis('off')
            
            
            plt.subplot(1, 3, 3)
            colors = ['red' if i == predicted_class else 'blue' for i in range(len(prediction))]
            bars = plt.bar(range(len(prediction)), prediction, color=colors, alpha=0.7)
            plt.title(f'Predição: {predicted_class_name}\nConfiança: {confidence:.2%}')
            plt.xlabel('Classe')
            plt.ylabel('Probabilidade')
            plt.xticks(range(len(prediction)))
            plt.grid(True, alpha=0.3)
            
            
            for bar, prob in zip(bars, prediction):
                if prob > 0.01:
                    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                             f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.show()
            
            return predicted_class, confidence, prediction
            
    except Exception as e:
        print(f"Erro ao processar imagem: {e}")
        return None, None, None

def batch_predict_external(model, folder_path, class_names, input_shape, n_channels=1):
    """
    Faz predição em todas as imagens de uma pasta
    """
    if not os.path.exists(folder_path):
        print(f"Pasta não encontrada: {folder_path}")
        return
    
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    
    image_files = []
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(os.path.join(folder_path, file))
    
    if not image_files:
        print(f"Nenhuma imagem encontrada em: {folder_path}")
        return
    
    print(f"\nProcessando {len(image_files)} imagens de: {folder_path}")
    print("="*60)
    
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] ", end="")
        pred_class, confidence, probs = predict_external_image(
            model, image_path, class_names, input_shape, n_channels
        )
        
        if pred_class is not None:
            results.append({
                'file': os.path.basename(image_path),
                'predicted_class': pred_class,
                'confidence': confidence,
                'class_name': class_names[pred_class] if pred_class < len(class_names) else f"Classe {pred_class}"
            })
    
    
    if results:
        print(f"\n{'='*60}")
        print("RESUMO FINAL DAS PREDIÇÕES")
        print(f"{'='*60}")
        for result in results:
            print(f"{result['file']} -> {result['class_name']} ({result['confidence']:.2%})")
        
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"\nEstatísticas:")
        print(f"   Média de confiança: {avg_confidence:.2%}")
        print(f"   Total de imagens processadas: {len(results)}")

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    
    x_train, y_train, x_test, y_test, n_channels, n_classes, info = load_data(DATASET_NAME)
    input_shape = (x_train.shape[1], x_train.shape[2], n_channels)
    
    
    class_names = get_class_names(DATASET_NAME, n_classes)
    
    print(f"\nDataset: {DATASET_NAME}")
    print(f"Classes: {n_classes} | Canais: {n_channels} | Imagens treino: {len(x_train)} | teste: {len(x_test)}")
    
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
    
    
    while True:
        print(f"\n{'='*60}")
        print("MENU DE PREDIÇÃO - CNN MEDMNIST")
        print(f"{'='*60}")
        print("1. Teste automático (dataset original)")
        print("2. Predição em imagem externa")
        print("3. Predição em lote (pasta com imagens)")
        print("4. Sair")
        
        choice = input("\nEscolha uma opção (1-4): ").strip()
        
        if choice == '1':
            
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            print(f"\nDesempenho no Teste:")
            print(f"   Acurácia: {test_acc:.4f}")
            print(f"   Loss: {test_loss:.4f}")
            
            
            sample_idx = np.random.randint(len(x_test))
            sample_pred = model.predict(x_test[sample_idx][np.newaxis,...], verbose=0)[0]
            predicted_class = np.argmax(sample_pred)
            
            real_class_name = class_names[y_test[sample_idx]]
            predicted_class_name = class_names[predicted_class]
            
            print(f"\nExemplo do Dataset:")
            print(f"   Real: {real_class_name}")
            print(f"   Previsto: {predicted_class_name}")
            print(f"   Confiança: {sample_pred[predicted_class]:.2%}")
            
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(x_test[sample_idx].squeeze(), cmap="gray" if n_channels==1 else None)
            plt.title(f"Real: {real_class_name}\nPrevisto: {predicted_class_name}")
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            plt.bar(range(n_classes), sample_pred)
            plt.title('Probabilidades')
            plt.xlabel('Classe')
            plt.ylabel('Probabilidade')
            plt.xticks(range(n_classes))
            plt.tight_layout()
            plt.show()
            
        elif choice == '2':
            
            image_path = input("\nDigite o caminho completo para a imagem: ").strip()
            if os.path.exists(image_path):
                predict_external_image(model, image_path, class_names, input_shape, n_channels)
            else:
                print("Arquivo não encontrado!")
                
        elif choice == '3':
            
            folder_path = input("\nDigite o caminho completo para a pasta com imagens: ").strip()
            batch_predict_external(model, folder_path, class_names, input_shape, n_channels)
            
        elif choice == '4':
            print("Saindo...")
            break
        else:
            print("Opção inválida! Escolha 1-4.")

if __name__ == '__main__':
    main()