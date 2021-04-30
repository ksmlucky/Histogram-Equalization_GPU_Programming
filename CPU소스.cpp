#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>

typedef unsigned char BYTE;
using namespace cv;

void histogram_equalization(Mat &img, Mat &out, int *hist, int *hist_sum, int height, int width);

void main()
{
	
	VideoCapture vcap("C:\\ǳ��1.avi");				//���� ��������

	namedWindow("Video1", WINDOW_NORMAL);			//���� �÷� ����
	namedWindow("Video2", WINDOW_NORMAL);			//�׷��� �����Ϸ� ��ȯ�� ����
	namedWindow("Video3", WINDOW_NORMAL);			//������׷� ��Ȱȭ�� ������ ����

	Mat maFrame;									//������ ĸ���� �̹���
	Mat img_gray;									//�׷��� �����Ϸ� ��ȯ�� �̹���

	vcap.read(maFrame);								//��ȯ�� �̹��� ��������

	int width, height;
	int *Histogram = new int[256];
	int *Histogram_Sum = new int[256];

	width = maFrame.cols;
	height = maFrame.rows;

	Mat output_hist(height, width, CV_8UC1);

	clock_t start = clock();						//���� �ð� ������ ���� �Լ�
	while (vcap.read(maFrame))						//�̹����� ������׷� ��Ȱȭ�� �����Ͽ� �ݺ� --> ����
	{
		cvtColor(maFrame, img_gray, CV_BGR2GRAY);	
		histogram_equalization(img_gray, output_hist, Histogram, Histogram_Sum, height, width);

		imshow("Video1", maFrame);					//������ ������ �� ���� ȭ�鿡 ���
		imshow("Video2", img_gray);
		imshow("Video3", output_hist);

		if (waitKey(25) >= 0)
			break;

		clock_t end = clock();

		printf("Elapsed time(CPU) = %u ms\n", end - start);	
	}
	

	vcap.release();
	destroyAllWindows();
}

void histogram_equalization(Mat &img, Mat &out, int *hist, int *hist_sum, int height, int width)
{
	int sum = 0;

	//������׷� ���� �迭 �ʱ�ȭ
	for (int i = 0; i < 256; i++)
	{
		hist[i] = 0;
		hist_sum[i] = 0;
	}

	//�Է� ������ ������׷� ���
	for (int i = 0; i < height; i++)						//�� �ȼ����� �´� ������ 1�� ������Ų��
	{											
		for (int j = 0; j < width; j++)
		{
			int value = img.at<uchar>(i, j);
			hist[value] += 1;
		}
	}

	//�Է� ������ ���� ������׷� ���
	for (int i = 0; i < 256; i++)
	{
		sum += hist[i];
		hist_sum[i] = sum;
	}

	//�Է� �׷��̽����� ������ ����ȭ�� ���� ������׷� ���
	float normalized_Histogram_Sum[256] = { 0.0, };
	int image_size = height * width;
	for (int i = 0; i < 256; i++)
	{																	
		normalized_Histogram_Sum[i] = hist_sum[i] / (float)image_size;
	}

	//������׷� ��Ȱȭ ����
	for (int i = 0; i < height; i++)	
	{									
		for (int j = 0; j < width; j++)	
		{
			out.at<uchar>(i, j) = (BYTE)(round(normalized_Histogram_Sum[img.at<uchar>(i, j)] * 255));
		}
	}
}


//���� �̹����� �ش��ϴ� �ȼ� ���� �ش��ϴ� normalize��Ų �迭��
//�׷��� ������ �̹����� �ִ� ũ���� 255�� ���Ͽ� �ش� ��ġ�� �ȼ�
//������ ��ġ�Ѵ�