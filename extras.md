# Extras

Em otimizadores, quanto maior o parâmetro $\beta$ (beta), mais suave a reta se torna.

Um dos problemas da sigmoide é que, como seu intervalo é [0, 1], caso o resultado seja 0 ou próximo de 0, pode ocorrer de que neurônios em camadas seguintes sejam desativados (visto que multiplicação por 0 resulta em 0).
Outro problema da função sigmoide é a rápida saturação, ou seja, a partir de certo ponto, a função não distingue bem entre valores, mesmo distantes.

Tangente hiperbólica resolve o problema de zerar camadas, visto que seu intervalo é de [-1, 1], porém, ainda possui o problema de saturação.

ReLU resolve o problema de saturação ignorando valores negativos.

LeakyReLU, ao invés de ignorar valores negativos, possui uma reta com um angulo menor que 45 graus para utilizar alguns valores negativos.

Parametric ReLU utiliza backpropagation para descobrir qual o melhor valor do ângulo da LeakyReLU.

MaxOut utiliza backpropagation para descobrir os melhores valores tanto para o ângulo da reta que define os valores negativos, quando da reta que define os valores positivos.
