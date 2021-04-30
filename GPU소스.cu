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

//CUDA���� ����ϱ� ���� __global__�Լ����� �����Ѵ�.
__global__ void histogram_equalization(BYTE *img, BYTE *out, int *hist, int *hist_sum, int height, int width);
__global__ void histogram_equalization2(BYTE *img, BYTE *out, int *hist, float *hist_sum, int height, int width);

void main()
{
	int i, j, y, x;

	Mat img, img_gray;

	VideoCapture vcap("C:\\ǳ��1.avi");
	vcap.read(img);

	namedWindow("INPUT", WINDOW_NORMAL);
	namedWindow("GRAY_SCALE", WINDOW_NORMAL);
	namedWindow("OUTPUT", WINDOW_NORMAL);

	//�̹��� ���� ����
	int width = img.cols;
	int height = img.rows;
	Mat output1(height, width, CV_8UC1);
	BYTE *img_buffer = new BYTE[height * width];
	BYTE *out_buffer1 = new BYTE[height * width];

	//�Է� �׷��̽����� ������ ������׷� ���
	float normalized_Histogram_Sum[256] = { 0.0, };
	int image_size = height * width;

	int *Histogram = new int[256];
	int *Histogram_Sum = new int[256];

	//GPU ����
	BYTE *dev_buffer1;
	BYTE *dev_img;

	//������׷� �迭 ����
	int *dev_Histogram = new int[256];
	int *dev_Histogram_Sum = new int[256];
	float *dev_normal = new float[256];

	cudaSetDevice(0);

	//GPU���� ����� Grid�� Block�� ũ�⸦ ����
	dim3 dimGrid((width - 1) / 32 + 1, (height - 1) / 32 + 1);
	dim3 dimBlock(32, 32);

	//Ŀ���Լ����� ����ϱ� ���� ����̽� �������� ����
	cudaMalloc((void **)&dev_img, sizeof(BYTE) * height * width);
	cudaMalloc((void **)&dev_buffer1, sizeof(BYTE) * height *width);
	cudaMalloc((void **)&dev_Histogram, sizeof(int) * 256);
	cudaMalloc((void **)&dev_Histogram_Sum, sizeof(int) * 256);
	cudaMalloc((void **)&dev_normal, sizeof(float) * 256);

	clock_t start = clock();

	//�̹����� ������׷� ��Ȱȭ�� �����Ͽ� �ݺ� --> ����
	while (vcap.read(img))
	{
		//������׷� ��źȭ ����

		//�׷��� ������ �������� ��ȯ
		cvtColor(img, img_gray, CV_BGR2GRAY);

		int sum = 0;

		for (int i = 0; i < 256; i++)
		{
			Histogram[i] = 0;
			Histogram_Sum[i] = 0;
		}

		//Mat �� �迭�� ��ȯ
		uchar* p;
		for (int j = 0; j < height; j++)
		{
			p = img_gray.ptr<uchar>(j);
			for (int i = 0; i < width; i++)
			{
				img_buffer[j * width + i] = p[i];
			}
		}
		
		//�޸�ī��
		cudaMemcpy(dev_img, img_buffer, sizeof(BYTE) * height * width, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_buffer1, out_buffer1, sizeof(BYTE) * height *width, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Histogram, Histogram, sizeof(int) * 256, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Histogram_Sum, Histogram_Sum, sizeof(int) * 256, cudaMemcpyHostToDevice);

		//�ϳ��� Ŀ�� �Լ��� ������ �� �ű⼭ ���� ������׷� ���� ���� ���� ������׷� ���� ���� ��
		//�� �ٸ� Ŀ�� �Լ��� �̹����� ����
		histogram_equalization << <dimGrid, dimBlock >> > (dev_img, dev_buffer1, dev_Histogram, dev_Histogram_Sum, height, width);

		cudaMemcpy(Histogram, dev_Histogram, sizeof(int) * 256, cudaMemcpyDeviceToHost);
		cudaMemcpy(Histogram_Sum, dev_Histogram_Sum, sizeof(int) * 256, cudaMemcpyDeviceToHost);

		//�Է� ������ ���� ������׷� ���
		for (int i = 0; i < 256; i++)
		{
			sum += Histogram[i];
			Histogram_Sum[i] = sum;
		}

		//�Է� �׷��̽����� ������ ����ȭ�� ���� ������׷� ���
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

//���� ó���� �� �� �� �ȼ� ���� �´� ������׷��� ���� Atomic ������ ���� ����
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

//������ ���� ������׷��� ���� �����Ǵ� �̹����� �ȼ��� ���� ����
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
