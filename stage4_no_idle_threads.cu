/*
Este exemplo mostra uma implementação ingênua de uma redução em CUDA.

A redução é feita em duas etapas. Primeiramenta reduzimos o array inteiro de 
tamanho 1 << 16 para um array de tamanho 256 com somas parciais.

Depois executamos a mesma redução sobre as somas parciais.

A posição acessada do array está vinculada ao índice da thread
(como aprendemos inicialmente).
*/
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>

using std::accumulate;
using std::generate;
using std::cout;
using std::vector;

#define TAMANHO_MEM_COMPARTILHADA 256 * 4

__global__ void reducaoSoma(int *v, int *v_reduzido) {
	// Aloca memória compartilhada para todos os kernels
    // threads estarão escrevendo nesse mesmo array durante 
    // a execução do codigo na GPU.
	__shared__ int somas_parciais[TAMANHO_MEM_COMPARTILHADA];

	// Cálculo do ID da thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Na primeira iteração, fazemos o primeiro passo
	// da redução. Calculamos um índice para além das threads,
	// e somamos com o índice em THREAD_ID (tid)
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Carrega o vetor resultante da primeira iteração da redução
	// na memória compartilhada.
	somas_parciais[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads(); // -> Sincronização: Necessário para prosseguir!

	// Vamos começar com um passo mais largo - 1/2 do tamanho do bloco
	// e dividindo o passo por 2 a cada iteração. 
	for (int passo = blockDim.x / 2; passo > 0; passo >>= 1) {
		// Cada thread realiza trabalho a não ser que esteja além do passo
		if (threadIdx.x < passo) {
			somas_parciais[threadIdx.x] += somas_parciais[threadIdx.x + passo];
		}
		__syncthreads();
	}

    // Deixamos, ao final de tudo, que a thread ID=0 
    // escreva o resultado da redução no primeiro elemento do array.
	if (threadIdx.x == 0) {
		v_reduzido[blockIdx.x] = somas_parciais[0];
	}
}

int main() {
	// Tamanho do array: 65536
	int N = 1 << 16;
	size_t bytes = N * sizeof(int);

	// Vetores (arrays) na maquina host: vetor original e
    // vetor reduzido.
	vector<int> host_arr(N);
	vector<int> host_arr_reduzido(N);

    // Inicializa o vetor , i. e., aloca memória na maquina host
    generate(begin(host_arr), end(host_arr), [](){ return rand() % 10; });

	// Aloca memória no dispositivo (device)
	int *device_arr, *device_arr_reduzido;
	cudaMalloc(&device_arr, bytes);
	cudaMalloc(&device_arr_reduzido, bytes);
	
	// Copia da maquina hospedeira (host) para o dispositivo (device)
	cudaMemcpy(device_arr, host_arr.data(), bytes, cudaMemcpyHostToDevice);
	
	// Tamanho do bloco em número de threads
	const int TAMANHO_BLOCO = 256;

	// Tamanho do grid em número de blocos
    // Vamos disparar somente metade do número de threads necessárias na primeira
	// iteração para não termos threads ociosas.
	int TAMANHO_GRID = N / TAMANHO_BLOCO / 2;

	// Faz duas chamadas para o kernel
	reducaoSoma<<<TAMANHO_GRID, TAMANHO_BLOCO>>>(device_arr, device_arr_reduzido);

	reducaoSoma<<<1, TAMANHO_BLOCO>>> (device_arr_reduzido, device_arr_reduzido);

	// Copia do dispositivo (device) para a máquina hospedeira (host)
	cudaMemcpy(host_arr_reduzido.data(), device_arr_reduzido, bytes, cudaMemcpyDeviceToHost);

	// Confere resultado
	assert(host_arr_reduzido[0] == std::accumulate(begin(host_arr), end(host_arr), 0));

	cout << "REDUÇÃO OCORREU COM SUCESSO.\n";

	return 0;
}