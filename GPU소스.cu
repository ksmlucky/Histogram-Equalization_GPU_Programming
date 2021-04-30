#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>

typedef unsigned char BYTE;
using namespace std;
using namespace cv;

Mat convertToMat(unsigned char *buffer, int width, int height);

//CUDA에서 사용하기 위한 __global__함수들을 선언한다.
__global__ void histogram_equalization(BYTE *img, BYTE *out, int *hist, int *hist_sum, int height, int width);
__global__ void histogram_equalization2(BYTE *img, BYTE *out, int *hist, float *hist_sum, int height, int width);

void main()
{
	int i, j, y, x;

	Mat img, img_gray;

	VideoCapture vcap("C:\\풍경1.avi");
	vcap.read(img);

	namedWindow("INPUT", WINDOW_NORMAL);
	namedWindow("GRAY_SCALE", WINDOW_NORMAL);
	namedWindow("OUTPUT", WINDOW_NORMAL);

	//이미지 가로 세로
	int width = img.cols;
	int height = img.rows;
	Mat output1(height, width, CV_8UC1);
	BYTE *img_buffer = new BYTE[height * width];
	BYTE *out_buffer1 = new BYTE[height * width];

	//입력 그레이스케일 영상의 히스토그램 계산
	float normalized_Histogram_Sum[256] = { 0.0, };
	int image_size = height * width;

	int *Histogram = new int[256];
	int *Histogram_Sum = new int[256];

	//GPU 설정
	BYTE *dev_buffer1;
	BYTE *dev_img;

	//히스토그램 배열 선언
	int *dev_Histogram = new int[256];
	int *dev_Histogram_Sum = new int[256];
	float *dev_normal = new float[256];

	cudaSetDevice(0);

	//GPU에서 사용할 Grid와 Block의 크기를 선언
	dim3 dimGrid((width - 1) / 32 + 1, (height - 1) / 32 + 1);
	dim3 dimBlock(32, 32);

	//커널함수에서 사용하기 위한 디바이스 변수들을 선언
	cudaMalloc((void **)&dev_img, sizeof(BYTE) * height * width);
	cudaMalloc((void **)&dev_buffer1, sizeof(BYTE) * height *width);
	cudaMalloc((void **)&dev_Histogram, sizeof(int) * 256);
	cudaMalloc((void **)&dev_Histogram_Sum, sizeof(int) * 256);
	cudaMalloc((void **)&dev_normal, sizeof(float) * 256);

	clock_t start = clock();

	//이미지에 히스토그램 평활화를 적용하여 반복 --> 영상
	while (vcap.read(img))
	{
		//히스토그램 평탄화 시작

		//그레이 스케일 영상으로 변환
		cvtColor(img, img_gray, CV_BGR2GRAY);

		int sum = 0;

		for (int i = 0; i < 256; i++)
		{
			Histogram[i] = 0;
			Histogram_Sum[i] = 0;
		}

		//Mat 을 배열로 전환
		uchar* p;
		for (int j = 0; j < height; j++)
		{
			p = img_gray.ptr<uchar>(j);
			for (int i = 0; i < width; i++)
			{
				img_buffer[j * width + i] = p[i];
			}
		}
		
		//메모리카피
		cudaMemcpy(dev_img, img_buffer, sizeof(BYTE) * height * width, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_buffer1, out_buffer1, sizeof(BYTE) * height *width, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Histogram, Histogram, sizeof(int) * 256, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Histogram_Sum, Histogram_Sum, sizeof(int) * 256, cudaMemcpyHostToDevice);

		//하나의 커널 함수를 진행한 후 거기서 나온 히스토그램 값을 통해 누적 히스토그램 값을 구한 후
		//또 다른 커널 함수로 이미지에 적용
		histogram_equalization << <dimGrid, dimBlock >> > (dev_img, dev_buffer1, dev_Histogram, dev_Histogram_Sum, height, width);

		cudaMemcpy(Histogram, dev_Histogram, sizeof(int) * 256, cudaMemcpyDeviceToHost);
		cudaMemcpy(Histogram_Sum, dev_Histogram_Sum, sizeof(int) * 256, cudaMemcpyDeviceToHost);

		//입력 영상의 누적 히스토그램 계산
		for (int i = 0; i < 256; i++)
		{
			sum += Histogram[i];
			Histogram_Sum[i] = sum;
		}

		//입력 그레이스케일 영상의 정규화된 누적 히스토그램 계산
		for (int i = 0; i < 256; i++)
		{
			normalized_Histogram_Sum[i] = Histogram_Sum[i] / (float)image_size;
		}

		cudaMemcpy(dev_normal, normalized_Histogram_Sum, sizeof(float) * 256, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Histogram_Sum, Histogram_Sum, sizeof(int) * 256, cudaMemcpyHostToDevice);

		histogram_equalization2 << <dimGrid, dimBlock >> > (dev_img, dev_buffer1, dev_Histogram, dev_normal, height, width);

		cudaMemcpy(out_buffer1, dev_buffer1, sizeof(BYTE) * width * height, cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		output1 = convertToMat(out_buffer1, width, height);
		
		imshow("INPUT", img);
		imshow("GRAY_SCALE", img_gray);
		imshow("OUTPUT", output1);

		if (waitKey(25) >= 0)
			break;

		clock_t end = clock();
		printf("Elapsed time(GPU) = %u ms\n", end - start);
	}
	
	cudaDeviceReset();

	cudaFree(dev_img);
	cudaFree(dev_buffer1);
	cudaFree(dev_Histogram);
	cudaFree(dev_Histogram_Sum);
	cudaFree(dev_normal);

	waitKey(0);
}

//예외 처리를 한 후 각 픽셀 값에 맞는 히스토그램의 값을 Atomic 연산을 통해 증가
__global__ void histogram_equalization(BYTE *img, BYTE *out, int *hist, int *hist_sum, int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ((i >= height) || (j >= width))
		return;
	int value = img[i * width + j];
	atomicAdd(&hist[value], 1);
	__syncthreads();
}

//구해진 누적 히스토그램을 통해 대응되는 이미지의 픽셀에 값을 적용
__global__ void histogram_equalization2(BYTE *img, BYTE *out, int *hist, float *hist_sum, int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ((i >= height) || (j >= width))
		return;

	out[i * width + j] = (BYTE)(round(hist_sum[img[i * width + j]] * 255));
	__syncthreads();
}

Mat convertToMat(unsigned char *buffer, int width, int height)
{
	Mat tmp(height, width, CV_8UC1);

	for (int x = 0; x < height; x++)
	{
		for (int y = 0; y < width; y++)
		{
			tmp.at<uchar>(x, y) = buffer[x * width + y];
		}
	}
	return tmp;
}
