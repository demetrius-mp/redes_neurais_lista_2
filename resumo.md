# Resumo

## Cross Entropy

> Não é necessário aplicar softmax antes de calcular a cross entropy pois a cross entropy já o faz internamente.

$V =$ vetor de saída da rede.

$i =$ posição da saída esperada

$\mathrm{CrossEntropy}(V, i) = -\ln\left(\frac{e^{V[i]}}{\sum_{v \in V} e^v}\right)$

## Convolução

$\mathrm{FeatureMap} = \lfloor\frac{D + (2P) - k}{S}\rfloor + 1$

$\mathrm{Params} = (k \times k \times in + 1) \times out$

$\mathrm{ParamsGroupConv} = (k \times k \times \frac{in}{groups} + 1) \times out$

> Depth-Wise Conv: a entrada é separada em canais, e cada canal da convolução é aplicada em um canal da entrada. Depois, junta os canais, e aplica uma Conv1x1. Para adicionar novos filtros, basta adicionar outra Conv1x1. Isso reduz a quantidade de parâmetros.

$\mathrm{ParamsDWConv} = (k \times k \times in + 1) + (in + 1) \times out$

### Exponential Moving Average (EMA)

> Faz suavização de uma série

$V_t = \beta V_{t-1} + (1 - \beta) \theta_t$

### Bias correction

> Faz com que valores iniciais da série tenham mais importância que os seguintes

$\hat{V} = \frac{V_t}{1 - \beta^t}$

### AdaGrad

> Aplica suavização no gradiente

$W_{t+1} = W_t - \frac{\alpha}{\sqrt{S_t + \epsilon}} \frac{\partial Loss}{\partial W_t}$

$S_t = S_{t-1} + (\frac{\partial Loss}{\partial W_t})^2$

### Adam

$W_{t+1} = W_t - \frac{\alpha}{\sqrt{S_t + \epsilon}} V_t$

> Momentum suaviza a loss com EMA:

$V_t = \beta_1 V_{t-1} + (1 - \beta_1) \frac{\partial Loss}{\partial W_t}$

> RMSProp suaviza o AdaGrad com EMA:

$S_t = \beta_2 S_{t-1} + (1 - \beta_2) (\frac{\partial Loss}{\partial W_t})^2$

### Weight Decay

> Tenta reduzir a complexidade do modelo diminuindo os pesos da rede

$w_j = w_j(1 - \alpha \lambda) - \alpha \frac{\partial Loss}{\partial w_j}$

### Batch normalization

- Corrige o problema das funções de ativações retornarem valores altos, utilizando média, desvio padrão, e score-z.
- Utiliza 2 parâmetros: gamma e beta, que são aprendidos pelo modelo
- Se gamma = desvio padrão, e beta = média, o algoritmo pode ignorar a normalização

> Média do batch

$\mu_B = \frac{1}{m} \displaystyle\sum_{i=1}^mx_i$

> Variância do batch

$\sigma_B^2 = \frac{1}{m} \displaystyle\sum_{i=1}^m (x_i - \mu_B)^2$

> Normalização

$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$

> Scale and shift

$y_i = \gamma \hat{x}_i + \beta$

### Técnicas de amostragem

- Estratificação: mantém a proporção em cada conjunto de dados. Não é necessário se a amostra é grande.
- Tamanho: quanto maior a amostra, mais próxima da distribuição real, mais fácil de capturar as particularidades.
- **O conjunto de teste deve ser representativo!!!**

### Holdout

- Utilizado em amostras grandes.
- Divide os dados em 2 partes, treinamento e teste.
- Se a amostra é grande, cada conjunto é definido com exemplos aleatórios.
- Se a amostra é pequena, o conjunto de testes pode não ser representativo.

### Cross validation

- Utilizado em amostras menores, muito caro
- Divide os dados em partições, onde uma partição será utilizada para teste, e o restante para treino.
- A quantidade de execuções é igual a quantidade de partições.
- O resultado é a média e desvio padrão da taxa de acerto em cada partição. O desvio padrão indica a estabilidade da amostra.
- Se a distribuição das classes é balanceado, recomenda-se que cada partição tenha ao menos 30 exemplos. Do contrário, devem ser maiores.

### Leave one out

- Instância específica do cross validation
- Número de partições é igual ao número de exemplos
- Quantidade de execuções igual a quantidade de exemplos.

### Nested cross validation

- Utilizada para fazer ajuste de parâmetros.
- O loop interno faz a seleção de parâmetros, e o externo verifica a qualidade do melhor modelo encontrado no loop interno.

### Ajuste de parâmetros

- Dividir a amostra em 3 partes: treino, validação, e teste.
- Validação é utilizada para ajuste de parâmetros.

### Treino, validação, e teste

- O erro no conjunto de teste é o que reflete melhor o desempenho do algoritmo.
- O conjunto de validação é utilizado várias vezes pelo algoritmo para fazer ajuste de parâmetros.
- O conjunto de teste é utilizado apenas uma vez, para medir o desempenho do algoritmo.

### Skip connection

- Faz com que a rede possa ignorar algumas camadas, somando o valor de entrada inicial (identidade), com a saída da convolução.
- Caso os tamanhos sejam incompatíveis, é aplicado downsample na identidade, ou seja, faz uma convolução para que tenha tamanho compatível com a saída

### Bottle neck

- Diminui bastante o número de parâmetros da rede, aplicando uma Conv1x1 antes da convolução desejada.
- Faz com que a rede tente generalizar a informação, utilizando menos canais.

### Squeeze and excitation

- Pondera os canais.

### Tokenização

- Palavras pouco frequentes no WordPiece são tokenizadas por caractere, enquanto mais frequentes são tokenizadas como uma palavra inteira.
- O CBOW retorna word embeddings com contextos similares, não significado.

### Geral

- A função logística satura rapidamente, devido ao seu curto intervalo, além de incentivar a propagação de desativação de camadas da rede.
