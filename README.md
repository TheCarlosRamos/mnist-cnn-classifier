# MNIST CNN Classifier

Este projeto implementa um classificador de dígitos manuscritos usando uma Rede Neural Convolucional (CNN) com TensorFlow/Keras, treinado no dataset MNIST.

## Requisitos
- Python 3.8+
- numpy
- matplotlib
- tensorflow (ou tensorflow-cpu)
- scikit-learn

## Instalação
Recomenda-se o uso de ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib tensorflow scikit-learn
```

## Como rodar

1. Certifique-se de que o arquivo `mnist_784.csv` está na mesma pasta do script.
2. Execute:

```bash
python3 exemplo_cnn_mnist.py
```

O script irá treinar o modelo (ou carregar um modelo salvo), avaliar no conjunto de teste e mostrar métricas e exemplos de predição.

## Sobre o dataset
O arquivo `mnist_784.csv` deve conter as imagens do MNIST, onde cada linha representa uma imagem (784 pixels) seguida do rótulo (0-9).

## Autor
https://github.com/TheCarlosRamos