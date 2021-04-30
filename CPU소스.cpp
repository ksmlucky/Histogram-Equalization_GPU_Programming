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
	
	VideoCapture vcap("C:\\풍경1.avi");				//영상 가져오기

	namedWindow("Video1", WINDOW_NORMAL);			//원본 컬러 영상
	namedWindow("Video2", WINDOW_NORMAL);			//그레이 스케일로 변환한 영상
	namedWindow("Video3", WINDOW_NORMAL);			//히스토그램 평활화를 적용한 영상

	Mat maFrame;									//영상을 캡쳐한 이미지
	Mat img_gray;									//그레이 스케일로 변환한 이미지

	vcap.read(maFrame);								//변환할 이미지 가져오기

	int width, height;
	int *Histogram = new int[256];
	int *Histogram_Sum = new int[256];

	width = maFrame.cols;
	height = maFrame.rows;

	Mat output_hist(height, width, CV_8UC1);

	clock_t start = clock();						//실행 시간 측정을 위한 함수
	while (vcap.read(maFrame))						//이미지에 히스토그램 평활화를 적용하여 반복 --> 영상
	{
		cvtColor(maFrame, img_gray, CV_BGR2GRAY);	
		histogram_equalization(img_gray, output_hist, Histogram, Histogram_Sum, height, width);

		imshow("Video1", maFrame);					//각각의 영상을 한 번에 화면에 출력
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

	//히스토그램 관련 배열 초기화
	for (int i = 0; i < 256; i++)
	{
		hist[i] = 0;
		hist_sum[i] = 0;
	}

	//입력 영상의 히스토그램 계산
	for (int i = 0; i < height; i++)						//각 픽셀값에 맞는 값들을 1씩 증가시킨다
	{											
		for (int j = 0; j < width; j++)
		{
			int value = img.at<uchar>(i, j);
			hist[value] += 1;
		}
	}

	//입력 영상의 누적 히스토그램 계산
	for (int i = 0; i < 256; i++)
	{
		sum += hist[i];
		hist_sum[i] = sum;
	}

	//입력 그레이스케일 영상의 정규화된 누적 히스토그램 계산
	float normalized_Histogram_Sum[256] = { 0.0, };
	int image_size = height * width;
	for (int i = 0; i < 256; i++)
	{																	
		normalized_Histogram_Sum[i] = hist_sum[i] / (float)image_size;
	}

	//히스토그램 평활화 적용
	for (int i = 0; i < height; i++)	
	{									
		for (int j = 0; j < width; j++)	
		{
			out.at<uchar>(i, j) = (BYTE)(round(normalized_Histogram_Sum[img.at<uchar>(i, j)] * 255));
		}
	}
}


//원래 이미지에 해당하는 픽셀 값에 해당하는 normalize시킨 배열에
//그레이 스케일 이미지의 최대 크기인 255씩 곱하여 해당 위치의 픽셀
//값으로 대치한다