#include <iostream>
#include <utility>

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

int main(){

    int a[] = {3, 7, 4, 8, 6, 2, 1, 5};
    int N = sizeof(a)/sizeof(a[0]);

    bitonicSortCpu(a, N, true);

    for(int i = 0; i<N; i++){
        std::cout<<a[i]<< " ";
    }

    return 0;
}