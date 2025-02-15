# Hackathon FIAP VisionGuard

## O Problema

A FIAP VisionGuard, empresa especializada em monitoramento de câmeras de segurança, busca otimizar seu software para identificar situações que possam representar riscos à segurança de estabelecimentos e comércios. Um dos desafios enfrentados é a detecção de objetos cortantes, como facas e tesouras, que podem ser utilizados em situações perigosas. Atualmente, não há um mecanismo automatizado eficaz para essa identificação, o que pode comprometer a segurança e a capacidade de resposta rápida das equipes responsáveis.

## A Proposta do Desafio

O objetivo do desafio é desenvolver um MVP (Minimum Viable Product) utilizando Inteligência Artificial para detecção supervisionada de objetos cortantes. Isso envolve a criação de um modelo capaz de identificar facas, tesouras e outros itens similares a partir de imagens capturadas por câmeras de segurança, emitindo alertas automáticos para uma central de monitoramento.

Para isso, os participantes devem:

- Construir ou selecionar um dataset contendo imagens de objetos cortantes em diferentes condições (ângulos, iluminação, etc.);
- Anotar corretamente o dataset, incluindo imagens negativas (sem objetos perigosos) para minimizar falsos positivos;
- Treinar um modelo supervisionado para identificação desses objetos;
- Desenvolver um sistema de alerta (por exemplo, via e-mail) para notificar a equipe de segurança quando um objeto for detectado.

## O Que Esperamos Como Entregável

Os participantes devem apresentar os seguintes itens como parte da entrega final do desafio:

- Documentação detalhada explicando o fluxo de desenvolvimento da solução, incluindo as tecnologias utilizadas, processos de treinamento do modelo e estratégia de detecção e alerta;
- Vídeo explicativo de até 15 minutos demonstrando a solução desenvolvida e seu funcionamento;
- Repositório no GitHub contendo todo o código-fonte, datasets utilizados (ou referência para eles) e instruções de uso e instalação.

## Dataset

O dataset utilizado pode ser obtido executando o arquivo [fiftyone_coco_filter.py](fiftyone_coco_filter.py).

Também realizamos testes bem-sucedidos com o dataset que pode ser baixado pelo arquivo [kaggle.py](kaggle.py) (não foi o usado no exemplo).

## Descrição

Esse projeto realiza a análise de um vídeo buscando identificar objetos cortantes, para disparar um e-mail para a equipe de segurança poder analisar.

## Pré-requisitos

- Python 3.x
- Bibliotecas listadas em `requirements.txt`
- Atenção, versão do torch pode precisar ser alterada de acordo com a versão do nvidia cuda instalada no dispositivo que está rodando o projeto

## Instalação

1. Clone o repositório:

    ```bash
    git clone https://github.com/Guilhermemzlima/FIAP-VisionGuard-Grupo-15
    cd FIAP-VisionGuard-Grupo-15
    ```

2. Instale as dependências:

    ```bash
    pip install -r requirements.txt
    ```

## Como Executar

## Dataset

O dataset utilizado pode ser obtido executando o arquivo [fiftyone_coco_filter.py](fiftyone_coco_filter.py).

 - Também realizamos testes bem-sucedidos com o dataset que pode ser baixado pelo arquivo [kaggle.py](kaggle.py) (não foi o usado no exemplo).


### Treinamento do Modelo de Anomalia

Antes de executar o projeto, é necessário treinar o modelo. Para isso, devemos executar o arquivo [train.py](model/train.py).

Execute o script de treinamento:

```bash
python model/train.py
```

O modelo será salvo na pasta raiz do projeto com o nome de yolo11l.pt

O processamento pode demorar bastante tempo, dependendo do hardware disponível.

### Rodando detecção
Com o nosso MVP estamos analisando um video, e com um client smtp, enviamos um e-mail quando identificamos um objeto cortantes. dito isso, precisamos configurar nossas variaveis de ambiente pra rodar o proximo passo

#### Na raiz do projeto crie um arquivo .env com as seguintes variaveis

```
SMTP_USER=
SMTP_PASSWORD=
SMTP_SEND=
VIDEO_PATH=
```
#### Sendo elas:
- SMTP_USER = email que vai enviar o e-mail
- SMTP_PASSWORD = senha do email que vai enviar o e-mail
- SMTP_SEND = email que vai receber o e-mail
- VIDEO_PATH = caminho do video que vai ser analisado


depois de configurar o env, basta executar o arquivo [detect.py](model/detect.py) para rodar a detecção

```bash
python model/detect.py
```


Assim, o e-mail configurado no .env deve receber um email com a imagem do objeto detectado e um texto avisando:
-  "Foi detectado um objeto cortante pela câmera de segurança. Verifique imediatamente."
- Estrutura dos Módulos
- model/train.py: Script para treinar o modelo YOLOv8.
- model/detect.py: Script para detectar objetos cortantes em um vídeo e enviar alertas por e-mail.
- alert/email_alert.py: Função para enviar alertas por e-mail.
- configs/data.yaml: Configuração do dataset para treinamento.
- .env: Arquivo de configuração das variáveis de ambiente.