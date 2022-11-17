# Fórmulas

## Cross-Entropy

$V =$ vetor de saída da rede.

$i =$ posição da saída esperada

$\mathrm{CrossEntropy}(V, i) = -\ln\left(\frac{e^{V[i]}}{\sum_{v \in V} e^v}\right)$

### Exemplo

Problema de classifição com 3 classes: `[positivo, neutro e negativo]`.

Vetor de saída da rede: `[0.5, 0, 1]`

Saída esperada: `positivo`.

Com isso temos que:

$V = [0.5, 0, 1]$, e $i = 1$, visto que a saída esperada é `positivo`, e a posição de `positivo` no vetor de classes é $1$.

Aplicando a Cross Entropy temos:

$\mathrm{CrossEntropy}([0.5, 0, 1], 1) = -\ln\left(\frac{e^{0.5}}{e^{0.5} + e^{0} + e^{1}}\right)$
