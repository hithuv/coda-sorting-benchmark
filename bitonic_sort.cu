#include <iostream>
#include <chrono>
#include <utility>
#include <fstream> 


//The below two functions are the recursive functions for CPU_Bitonic, but later
// void bitonicMerge(int* a, int N, int ind, bool dir){
//     if(N==1)return;
//     for(int i = ind; i<ind+N/2; i++){
//         if(dir == a[i]>a[i+N/2]){
//             std::swap(a[i], a[i+N/2]);
//         }
//     }
//     bitonicMerge(a, N/2, ind, dir);
//     bitonicMerge(a, N/2, ind+N/2, dir);
// }

// void bitonicSort(int* a, int N, int ind, bool dir){
//     if(N == 1)return;
//     bitonicSort(a, N/2, ind, true);
//     bitonicSort(a, N/2, ind+N/2, false);
//     bitonicMerge(a, N, ind, dir);
// }

void bitonicSortCpu(int* a, int N, bool dir){
    for (int k = 2; k<=N; k*=2){
        for(int j=k/2; j>0; j/=2){
            for(int i = 0; i<N; i++){
                int partner = i ^ j; 
                if(partner > i && partner<N){
                    bool dir = ((i&k) == 0);
                    if(dir == a[i]>a[partner]){
                        std::swap(a[i], a[partner]);
                    }
                }
            }
        }
    }
}

__global__
void bitonicSortKernel(int *a, int k, int j, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    int partner = i ^ j; 
    if(partner > i && partner<N){
        bool dir = ((i&k) == 0);
        if(dir == a[i]>a[partner]){
            // a[partner]= a[partner]+a[i];
            // a[i] = a[partner]-a[i];
            // a[partner] = a[partner] - a[i];
            //the above way of swapping is not thread safe
            int temp = a[i];
            a[i] = a[partner];
            a[partner] = temp;
        }
    } 
}

void bitonicSortGpu(int *d_a, int N){
    int threadsPerBlock = 256;
    int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;
    for (int k = 2; k<=N; k*=2){
        for(int j=k/2; j>0; j/=2){
            bitonicSortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, k, j, N);
        }
    }
}

__device__
int lowerbound(int *d_input, int width, int value){
    int l = 0;
    int r = width-1;
    while(l<=r){
        int mid = (r-l)/2+l;
        if(d_input[mid]<=value){
            l = mid+1;
        }
        else r = mid-1;
    }
    return l;
}

__global__
void mergeSortKernel(int *d_input, int* d_output, int N, int width){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N){
        return;
    }

    int a_start = (i/(width*2))*(width*2);
    int a_end = a_start+width; //a is until a_end (excluding)
    int b_start = a_end;
    // int b_end = b_start+width;

    int curr_value = d_input[i];
    int rank;
    if(i<a_end){
        rank = i-a_start + lowerbound(d_input+b_start, width, curr_value);
    }
    else{
        rank = i-b_start + lowerbound(d_input+a_start, width, curr_value);
    }

    d_output[a_start + rank] = curr_value;


}

void MergeSortGPU(int *d_a, int N){
    int *d_temp;
    cudaMalloc(&d_temp, N*sizeof(int));

    int *d_input = d_a;
    int *d_output = d_temp;

    int threadsPerBlock = 256;
    int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;

    for(int k = 1; k<N; k*=2){
        mergeSortKernel<<<blocksPerGrid,threadsPerBlock>>>(d_input, d_output, N, k);
        cudaDeviceSynchronize();

        std::swap(d_input, d_output); // since now output contains the input for next iter
    }
    if (d_input != d_a) {
        cudaMemcpy(d_a, d_input, N * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_temp);

}

int main(){

    std::ofstream outputFile("results.csv");
    outputFile << "N,CPU,GPU_Bitonic,GPU_Bitonic_w/mem,GPU_MergeSort" << std::endl;

    for(int n = 3; n <= 28; n++){
        int N = 1<<n;
        std::cout << "n = "<<n<<std::endl;
        int* h_a_cpu = new int[N]; //h_a means host_array
        int* h_a_gpu_1 = new int[N]; 
        int* h_a_gpu_2 = new int[N]; 

        for(int i = 0; i<N; i++){
            int random_number = rand()%RAND_MAX;
            h_a_cpu[i] = random_number;
            h_a_gpu_1[i] = random_number;
            h_a_gpu_2[i] = random_number;
            // std::cout<<random_number<<" ";
        }
        // std::cout<<std::endl;

        //CPU Profile - just this one func below
        auto start_cpu = std::chrono::high_resolution_clock::now();
        // bitonicSortCpu(h_a_cpu, N, true);
        auto end_cpu = std::chrono::high_resolution_clock::now();



        //GPU Profile for Bitonic Sort 
        cudaEvent_t start1, start2, stop1, stop2;
        cudaEventCreate(&start1);
        cudaEventCreate(&start2);
        cudaEventCreate(&stop1);
        cudaEventCreate(&stop2);
        cudaEventRecord(start1, 0); // 0 for default stream
        int* d_a_1; //d_a means device_array. Host is CPU, Device is GPU. 
        cudaMalloc(&d_a_1, N*sizeof(int));

        cudaMemcpy(d_a_1, h_a_gpu_1, N*sizeof(int), cudaMemcpyHostToDevice);
        cudaEventRecord(start2, 0); // 0 for default stream
        
        bitonicSortGpu(d_a_1, N);
        
        cudaDeviceSynchronize();
        cudaEventRecord(stop2, 0);
        cudaEventSynchronize(stop2);
        cudaMemcpy(h_a_gpu_1, d_a_1, N*sizeof(int), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop1, 0);
        cudaEventSynchronize(stop1);

        
        cudaFree(d_a_1);
        auto elapsed_cpu_s = end_cpu - start_cpu;
        long long elapsed_cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_cpu_s).count();
        std::cout << "CPU Elapsed time: " << elapsed_cpu_ms << " ms\n";

        float elapsedTime1, elapsedTime2;
        cudaEventElapsedTime(&elapsedTime1, start1, stop1);
        cudaEventElapsedTime(&elapsedTime2, start2, stop2);
        std::cout << "GPU_memory: " << elapsedTime1 << " ms\n" << std::endl;
        std::cout << "GPU_w/memory: " << elapsedTime2 << " ms\n" << std::endl;


        

        //GPU Profile for Merge Sort
        cudaEvent_t start3, stop3;
        cudaEventCreate(&start3);
        cudaEventCreate(&stop3);
        cudaEventRecord(start3, 0); // 0 for default stream
        int *d_a_2;
        cudaMalloc(&d_a_2, N*sizeof(int));
        cudaMemcpy(d_a_2, h_a_gpu_2, N*sizeof(int), cudaMemcpyHostToDevice);

        MergeSortGPU(d_a_2, N);

        cudaMemcpy(h_a_gpu_2, d_a_2, N*sizeof(int), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop3, 0); // 0 for default stream
        cudaEventSynchronize(stop3);
        cudaFree(d_a_2);

        float elapsedTime3;
        cudaEventElapsedTime(&elapsedTime3, start3, stop3);
        std::cout << "GPU_merge_sort: " << elapsedTime3 << " ms\n" << std::endl;

        delete[] h_a_cpu;
        delete[] h_a_gpu_1;
        delete[] h_a_gpu_2;
        
        
        outputFile<<n<<","<<elapsed_cpu_ms<<","<<elapsedTime1<<","<<elapsedTime2<<","<<elapsedTime3<<std::endl;

    }
    return 0;
}
