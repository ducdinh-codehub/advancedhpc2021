#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;
    bool is_shared = false;
    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }
    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out-new.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            
            labwork.labwork5_GPU(is_shared);
            if(is_shared==false){
                labwork.saveOutputImage("labwork5-gpu-out-no-shared.jpg");
            }else{
                labwork.saveOutputImage("labwork5-gpu-out-shared.jpg");
            }
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-outjpg");
            labwork.saveOutputImage("/tmp/labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    printf("Open MP");
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    // do something here
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        # pragma omp parallel for
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] + (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++){
        // get informations from individual device
        printf("------------------------------------\n");
	cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        // something more here
	printf("Name: %s\n", prop.name);
	printf("Clock rate: %d\n", prop.clockRate);
	printf("Core count: %d\n", getSPcores(prop));
	printf("Multiprocessors: %d\n", prop.multiProcessorCount);
	printf("Warp size: %d\n", prop.warpSize);
	printf("Memory info\n");
	printf("Memory clock rate: %d\n", prop.memoryClockRate);
	printf("Memory bus width: %d\n", prop.memoryBusWidth);
    }

}
__global__ void grayscale(uchar3 *input, uchar3 *output) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x; 
	output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
        output[tid].z = output[tid].y = output[tid].x;
}
void Labwork::labwork3_GPU() {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;	
    //char *hostInput = inputImage->buffer; // Perfect version
    char *hostInput = (char*) malloc(inputImage->width * inputImage->height * 3); // Test version
    char *hostOutput = new char[inputImage->width * inputImage->height * 3]; // Test version
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it$
        # pragma omp parallel for
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] + (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }

    // Allocate CUDA memory    
    uchar3 *devInput;
    uchar3 *devOutput;
    //cudaMalloc(&devInput, pixelCount*3); // Perfect version
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3)); // Test version
    //cudaMalloc(&devOutput, pixelCount*3); // Perfect version
    cudaMalloc(&devOutput, pixelCount * sizeof(float)); // Test version
    
    // Copy CUDA Memory from CPU to GPU
    //cudaMemcpy(devInput, hostInput, pixelCount*3, cudaMemcpyHostToDevice); // Perfect version
    cudaMemcpy(devInput, hostInput, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice); // Test version


    // Processing
    int blockSize = 64;
    int nBlock = pixelCount/blockSize;
    grayscale<<<nBlock, blockSize>>>(devInput, devOutput);

    // Copy CUDA Memory from GPU to CPU
    //cudaMemcpy(outputImage, devOutput, pixelCount*3, cudaMemcpyDeviceToHost); // Perfect version 
    cudaMemcpy(hostOutput, devOutput, pixelCount*sizeof(float), cudaMemcpyDeviceToHost); // Test version

    // Cleaning
    //free(hostInput);
    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void grayscale_2d(uchar3 *input, uchar3 *output, int img_width, int img_height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col >= img_width || row >= img_height) return;
    int tid = row * img_width + col;
    output[tid].x = (char)(((int)input[tid].x + (int)input[tid].y + (int)input[tid].z) / 3);
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork4_GPU() {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    char *hostInput = inputImage->buffer;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory    
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);
    
    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, hostInput, pixelCount * 3, cudaMemcpyHostToDevice);

    // Processing
    dim3 blockSize = dim3(32,32);
    dim3 gridSize = dim3((int)((inputImage->width + blockSize.x - 1) / blockSize.x), (int)((inputImage->height + blockSize.y - 1)/ blockSize.y));
    grayscale_2d<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    // Cleaning
    free(hostInput);
    cudaFree(devInput);
    cudaFree(devOutput);
}

//void Labwork::labwork5_GPU(){
//}

float gaussianBlur[7][7]={
        {0,   0,   1,   2,   1,   0,   0},
        {0,   3,   13,  22,  13,  3,   0},
        {1,   13,  59,  97,  59,  13,  1},
        {2,   22,  97,  159, 97,  22,  2},
        {1,   13,  59,  97,  59,  13,  1},
        {0,   3,   13,  22,  13,  3,   0},
        {0,   0,   1,   2,   1,   0,   0}
};

/*
float gaussianBlur[]={
        0,   0,   1,   2,   1,   0,   0,
        0,   3,   13,  22,  13,  3,   0,
        1,   13,  59,  97,  59,  13,  1,
        2,   22,  97,  159, 97,  22,  2,
        1,   13,  59,  97,  59,  13,  1,
        0,   3,   13,  22,  13,  3,   0,
        0,   0,   1,   2,   1,   0,   0
};*/

/*
float gaussianBlur[7][7] =
        [[ 0, 0, 1, 2, 1, 0, 0 ],
        [ 0, 3, 13, 22, 13, 3, 0 ],
        [ 1, 13, 59, 97, 59, 13, 1 ],
        [ 2, 22, 97, 159, 97, 22, 2 ],
        [ 1, 13, 59, 97, 59, 13, 1 ],
        [ 0, 3, 13, 22, 13, 3, 0 ],
        [ 0, 0, 1, 2, 1, 0, 0 ]];*/
void Labwork::labwork5_CPU(){
        int pixelCount = inputImage->width * inputImage->height;
        outputImage = static_cast<char *>(malloc(pixelCount * 3));
        for (int row = 3; row < inputImage->height-3; row++) {
            for (int col = 3; col < inputImage->width-3; col++) {
                int sumR = 0;
                int sumG = 0;
                int sumB = 0;
                for (int j = 0; j < 7; j++) {
                    for (int i = 0; i < 7; i++) {
                        int pos = (col - i - 3) + (row - j - 3) * inputImage->width;
                        sumR += inputImage->buffer[pos * 3]*gaussianBlur[j][i];
                        sumG += inputImage->buffer[pos * 3 + 1]*gaussianBlur[j][i];
                        sumB += inputImage->buffer[pos * 3 + 2]*gaussianBlur[j][i];
                    }
                }
                sumR /= 1003;
                sumG /= 1003;
                sumB /= 1003;
                int pos = col+ row * inputImage->width;
                outputImage[pos * 3] = sumR;
                outputImage[pos * 3 + 1] = sumG;
                outputImage[pos * 3 + 2] = sumB;
            }
        }
}
__global__ void gaussianNoShared(uchar3 *input, uchar3 *output, int imgWidth, int imgHeight){
        float gaussianBlur[7][7]={
            {0,   0,   1,   2,   1,   0,   0},
            {0,   3,   13,  22,  13,  3,   0},
            {1,   13,  59,  97,  59,  13,  1},
            {2,   22,  97,  159, 97,  22,  2},
            {1,   13,  59,  97,  59,  13,  1},
            {0,   3,   13,  22,  13,  3,   0},
            {0,   0,   1,   2,   1,   0,   0}
        };
        int col = threadIdx.x + blockIdx.x + blockDim.x;
        int row = threadIdx.y + blockIdx.y + blockDim.y;
        int tid = row * imgWidth + col;
        int sumR = 0;
        int sumG = 0;
        int sumB = 0;
        for (int j = 0; j < 7; j++) {
            for (int i = 0; i < 7; i++) {
                int cell_id = tid + i + j * imgWidth;
                sumR += input[cell_id].x * gaussianBlur[j][i];
                sumG += input[cell_id].y * gaussianBlur[j][i];
                sumB += input[cell_id].z * gaussianBlur[j][i];
            }
        }
    
        output[tid].x = sumR/1003;
        output[tid].y = sumG/1003;
        output[tid].z = sumB/1003;
}
__global__ void gaussianShared(uchar3 *input, uchar3 *output, int imgWidth, int imgHeight){
        float gaussianBlur[7][7]={
            {0,   0,   1,   2,   1,   0,   0},
            {0,   3,   13,  22,  13,  3,   0},
            {1,   13,  59,  97,  59,  13,  1},
            {2,   22,  97,  159, 97,  22,  2},
            {1,   13,  59,  97,  59,  13,  1},
            {0,   3,   13,  22,  13,  3,   0},
            {0,   0,   1,   2,   1,   0,   0}
        };
        
        int col = threadIdx.x + blockIdx.x + blockDim.x;
        int row = threadIdx.y + blockIdx.y + blockDim.y;
        int tid = row * imgWidth + col;

        __shared__ float gb[7][7];
        if(threadIdx.x < 7 && threadIdx.y < 7)
            gb[threadIdx.x][threadIdx.y] = gaussianBlur[row][col];
        __syncthreads();

        int sumR = 0;
        int sumG = 0;
        int sumB = 0;
        for (int j = 0; j < 7; j++) {
            for (int i = 0; i < 7; i++) {
                int cell_id = tid + i + j * imgWidth;
                sumR += input[cell_id].x * gb[threadIdx.x][threadIdx.y];
                sumG += input[cell_id].y * gb[threadIdx.x][threadIdx.y];
                sumB += input[cell_id].z * gb[threadIdx.x][threadIdx.y];
            }
        }
    
        output[tid].x = sumR/1003;
        output[tid].y = sumG/1003;
        output[tid].z = sumB/1003;

}
void Labwork::labwork5_GPU(bool shared) {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;	

    // Allocate CUDA memory    
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount);
    cudaMalloc(&devOutput, pixelCount);
    
    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, inputImage, pixelCount, cudaMemcpyHostToDevice);

    // Processing
    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = ((int) ((inputImage->width + blockSize.x - 1)/blockSize.x), (int)((inputImage->height + blockSize.y - 1)/blockSize.y));
    if(shared == false){
        gaussianNoShared<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height);
    }else{
        gaussianShared<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height);
    }
    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(inputImage, devOutput, pixelCount, cudaMemcpyHostToDevice);

    // Cleaning
    //free(hostInput);
    cudaFree(devInput);
    cudaFree(devOutput);
}

__device__ int brightness(){

}
__device__ int blending(){

}

__device__ int binarization(int input, int threshold){
    if(input < threshold){
        return 0;
    }else{
        return 255;
    }
}

__global__ void labwork6_a(uchar3 *input, uchar3 *output, int imgWidth, int imgHeight){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if(col > imgHeight || row > imgWidth){
        return;
    }
    int tid = col + row * imgWidth;

    int threshold = 128;

    output[tid].x = binarization(input[tid].x, threshold);
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork6_GPU() {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;	

    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    // Allocate CUDA memory    
    uchar3 *devInput;
    uchar3 *devOutput;
    uchar3 *devGray;

    cudaMalloc(&devInput, pixelCount*3); // Perfect version
    cudaMalloc(&devOutput, pixelCount*3); // Perfect version
    cudaMalloc(&devGray, pixelCount*3); // Perfect version
    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount*3, cudaMemcpyHostToDevice); // Perfect version

    // Processing
    dim3 blockSize = dim3(16, 16);
    dim3 gridSize = dim3((int) ((inputImage->width + blockSize.x - 1)/blockSize.x), (int)((inputImage->height + blockSize.y - 1)/blockSize.y));

    grayscale_2d<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height);
    //labwork6_a<<<gridSize, blockSize>>>(devGray, devOutput, inputImage->width, inputImage->height);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devOutput, pixelCount*3, cudaMemcpyDeviceToHost); // Perfect version 

    // Cleaning
    
    cudaFree(devInput);
    cudaFree(devOutput);
    cudaFree(devGray);
}
__device__ char cal(char input, char max, int min){
    char grayStretch = (float)((input - min)/(max - min)) * 255;
    return grayStretch;
}

__global__ void find_max(char *in, char *out) {
    // dynamic shared memory size, allocated in host
    extern __shared__ char cache[];
    // cache the block content
    unsigned int blockSize = blockDim.x * blockDim.y;
    unsigned int localtid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = blockIdx.x + blockSize + localtid;
    cache[localtid] = in[tid];
    __syncthreads();
    // reduction in cache
    for (int s = 1; s < blockDim.x; s *= 2) {
        cache[localtid] = max(cache[localtid], cache[localtid + s]);
        __syncthreads();
    }
    // only first thread writes back
    if (localtid == 0) out[blockIdx.x] = cache[0];
}

__global__ void find_min(char *in, char *out) {
    // dynamic shared memory size, allocated in host
    extern __shared__ char cache[];
    // cache the block content
    unsigned int blockSize = blockDim.x * blockDim.y;
    unsigned int localtid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = blockIdx.x + blockSize + localtid;
    cache[localtid] = in[tid];
    __syncthreads();
    // reduction in cache
    for (int s = 1; s < blockDim.x; s *= 2) {
        cache[localtid] = min(cache[localtid], cache[localtid + s]);
        __syncthreads();
    }
    // only first thread writes back
    if (localtid == 0) out[blockIdx.x] = cache[0];
}

__global__ void Calculate(char *in, char *out, int imgWidth, int imgHeight, char *max, char *min){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if(col > imgHeight || row > imgWidth){
        return;
    }
    int tid = col + row * imgWidth;

    int threshold = 128;

    char grayStretch = cal(in[tid], max[0], min[0]);
    out[tid * 3] = out[tid * 3 + 1] = out[tid * 3 + 2] = grayStretch;
}

void Labwork::labwork7_GPU() {
    // Calculate number of pixels
    char* maxVar;
    char* minVar;

    int pixelCount = inputImage->width * inputImage->height;	

    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    // Allocate CUDA memory    
    char *devInput;
    char *devOutput;

    cudaMalloc(&devInput, pixelCount*3); // Perfect version
    cudaMalloc(&devOutput, pixelCount*3); // Perfect version

    cudaMemcpy(devInput, inputImage->buffer, pixelCount*3, cudaMemcpyHostToDevice); // Perfect version


    cudaMalloc(&maxVar, pixelCount*3);
    cudaMalloc(&minVar, pixelCount*3);
    // Copy CUDA Memory from CPU to GPU

    // Processing
    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((int) ((inputImage->width + blockSize.x - 1)/blockSize.x), (int)((inputImage->height + blockSize.y - 1)/blockSize.y));

    find_max<<<gridSize, blockSize, blockSize.x*blockSize.y>>>(devInput, maxVar);
    find_min<<<gridSize, blockSize, blockSize.x*blockSize.y>>>(devInput, minVar);
    Calculate<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height, maxVar, minVar);


    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devOutput, pixelCount*3, cudaMemcpyDeviceToHost); // Perfect version 

    // Cleaning
    
    cudaFree(devInput);
    cudaFree(devOutput);
    cudaFree(maxVar);
    cudaFree(minVar);
}

void Labwork::labwork8_GPU() {
}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU(){
}


























