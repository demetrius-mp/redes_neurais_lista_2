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

## Quantidade de parâmetros da Group Convolution

- $k =$ tamanho do kernel
- $in =$ # canais da entrada
- $out =$ # canais do kernel
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
