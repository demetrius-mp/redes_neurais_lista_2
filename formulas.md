# Fórmulas

## Cross Entropy

$V =$ vetor de saída da rede.

$i =$ posição da saída esperada

$\mathrm{CrossEntropy}(V, i) = -\ln\left(\frac{e^{V[i]}}{\sum_{v \in V} e^v}\right)$

### Exemplo Cross Entropy

Problema de classifição com 3 classes: `[positivo, neutro e negativo]`.

Vetor de saída da rede: `[0.5, 0, 1]`

Saída esperada: `positivo`.

Com isso temos que:

$V = [0.5, 0, 1]$, e $i = 1$, visto que a saída esperada é `positivo`, e a posição de `positivo` no vetor de classes é $1$.

Aplicando a Cross Entropy temos:

$\mathrm{CrossEntropy}([0.5, 0, 1], 1) = -\ln\left(\frac{e^{0.5}}{e^{0.5} + e^{0} + e^{1}}\right)$

## Convolução básica

- $k =$ tamanho do kernel
- $in =$ # canais da entrada
- $out =$ # canais do kernel

### Número de parâmetros de uma convolução

$\mathrm{Conv}(k, in, out) = (k \times k \times in + 1) \times out$

### Tamanho da feature map

- $F =$ dimensão da entrada
- $S =$ stride
- $P =$ padding

$\mathrm{FeatMap}(k, F, S, P) = \lfloor\frac{F + (2P) - k}{S}\rfloor + 1$

## Quantidade de parâmetros da Group Convolution

- $groups =$ # grupos

$\mathrm{GroupConv}(k, in, out, groups) = (k \times k \times \frac{in}{groups} + 1) \times out$

### Exemplo Group Convolution

As matrizes são lidas como `[altura, largura, canais]`

- Entrada: `[28, 28, 128]`
- Kernel: `[3, 3, 256]`
- Groups: `2`

Para efeito de comparação, o número de parâmetros utilizando convolução normal é o seguinte:

$(3 \times 3 \times 128 + 1) \times 256 = 295168$

O número de parâmetros utilizando GroupConv é o seguinte:

$(3 \times 3 \times \frac{128}{2} + 1) \times 256 = 147712$

## Quantidade de parâmetros da DepthWise Convolution

- $k =$ tamanho do kernel
- $in =$ # canais da entrada
- $out =$ # canais do kernel

$\mathrm{DWConv}(k, in, out) = (k \times k \times in + 1) + (in + 1) \times out$

### Exemplo DepthWise Convolution

As matrizes são lidas como `[altura, largura, canais]`

- Entrada: `[128, 128, 3]`
- Kernel: `[3, 3, 64]`

Para efeito de comparação, o número de parâmetros utilizando convolução normal é o seguinte:

$(3 \times 3 \times 3 + 1) \times 64 = 1792$

O número de parâmetros utilizando DWConv é o seguinte:

$(3 \times 3 \times 3 + 1) + (3 + 1) \times 64 = 284$

## Otimizadores

### Exponential Moving Average (EMA)

> Faz suavização de uma série

$V_t = \beta V_{t-1} + (1 - \beta) \theta_t$

### Bias correction

> Faz com que valores iniciais da série tenham mais importância que os seguintes

$\hat{V} = \frac{V_t}{1 - \beta^t}$

### Momentum

> Aplica suavização na Loss utilizando EMA

$W_{t+1} = W_t - \alpha V_t$

$V_t = \beta V_{t-1} + (1 - \beta) \frac{\partial Loss}{\partial W_t}$

### AdaGrad

> Aplica suavização no gradiente

$W_{t+1} = W_t - \frac{\alpha}{\sqrt{S_t + \epsilon}} \frac{\partial Loss}{\partial W_t}$

$S_t = S_{t-1} + (\frac{\partial Loss}{\partial W_t})^2$

### RMSProp

> Aplica suavização no AdaGrad utilizando EMA

$W_{t+1} = W_t - \frac{\alpha}{\sqrt{S_t + \epsilon}} \frac{\partial Loss}{\partial W_t}$

$S_t = \beta S_{t-1} + (1 - \beta) (\frac{\partial Loss}{\partial W_t})^2$

### Adam

> Mistura entre Momentum e RMSProp

$W_{t+1} = W_t - \frac{\alpha}{\sqrt{S_t + \epsilon}} V_t$

$V_t = \beta_1 V_{t-1} + (1 - \beta) \frac{\partial Loss}{\partial W_t}$

$S_t = \beta_2 S_{t-1} + (1 - \beta) (\frac{\partial Loss}{\partial W_t})^2$

### Weight Decay

> Tenta reduzir a complexidade do modelo diminuindo os pesos da rede

$w_j = w_j(1 - \alpha \lambda) - \alpha \frac{\partial Loss}{\partial w_j}$

## Amostragem

A amostra tem que representar a distribuição dos dados do mundo real, ou seja, que serão utilizados pelo modelo em produção.

Como obter uma amostra representativa?

- Estratificação: manter a mesma proporção em cada conjunto de dados. Se o conjunto é grande, não é necessário utilizar estratificação.
- Tamanho da amostra: quanto maior, mais próxima da distribuição real. Grande o suficiente para capturar as particularidades da distribuição.
- Proporção: deve manter a mesma proporção da distribuição real.

> A pergunta principal é: o conjunto de teste é representativo?

### Holdout (Amostras grandes)

A distribuição é dividida em 2 partes, treinamento e teste. Caso a amostra seja grande, os  exemplos de cada conjuntos são escolhidos aleatoriamente.

O problema do holdout, é que se a amostra é pequena, o conjunto de teste não é representativo.

### Cross Validation

O conjunto é dividido em *folds* (partições), onde um fold será utilizado para teste, e o restante para treino. Isso é repetido para cada fold. A quantidade de execuções é igual a quantidade de partições.

O resultado da cross validation é a média e o desvio padrão da taxa de acerto em cada fold. O desvio padrão indica a estabilidade da amostra.

O cross validation é uma alternativa para caso não haja tantos exemplos para utilizar holdout.

Caso a distribuição dos dados seja balanceada, é recomendado que cada partição tenha pelo menos 30 exemplos. Caso contrário, as partições devem ser maiores.

### Leave one out

Instância específica do cross validation onde o número de partições é igual ao número de exemplos. Ou seja, a quantidade de execuções é igual a quantidade de exemplos.

### Ajuste de parâmetros

O mais comum é dividir a amostra em 3 partes: treino, validação, e teste.

A validação é utilizada para fazer ajuste de parâmetros.
