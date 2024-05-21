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

#define TAMANHO_MEM_COMPARTILHADA 256

__global__ void reducaoSoma(int *v, int *v_reduzido) {
	// Aloca memória compartilhada para todos os kernels
    // threads estarão escrevendo nesse mesmo array durante 
    // a execução do codigo na GPU.
	__shared__ int somas_parciais[TAMANHO_MEM_COMPARTILHADA];

	// Cálculo do ID da thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Carrega os primeiros elementos na memória compartilhada
	somas_parciais[threadIdx.x] = v[tid];
	__syncthreads(); // -> Sincronização: Necessário para prosseguir!

	// Aqui, cada thread vai somar 2 elementos, então deve
    // acessar outra posição além de 'tid' (thread ID).
    // Vamos criar um "passo" - que é o passo da thread.
	// Dessa vez, threads ativas sempre são sequenciais
	for (int passo = 1; passo < blockDim.x; passo *= 2) {
		// O índice será o ID das threads sequenciais
    	int index  = 2 * passo * threadIdx.x;

		// Cada thread faz trabalho a não ser que esteja além do índice
		// máximo possível dado o tamanho do bloco
		if (index < blockDim.x) {
			somas_parciais[index] += somas_parciais[index + passo];
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

	// Tamanho do grid em número de bloco
    // (tamanho do array / número de threads por bloco)
	int TAMANHO_GRID = N / TAMANHO_BLOCO;

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