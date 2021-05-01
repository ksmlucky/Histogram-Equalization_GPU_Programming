# Histogram-Equalization_GPU_Programming
CUDA를 이용한 병렬 프로그래밍으로 히스토그램 평활화 구현(영상에 적용)

- OpenCV를 이용하여 히스토그램 평활화를 진행할 영상을 가져오고, 반복문으로 영상을 히스토그램 평활화하고 그 실행시간을 측정해서 출력하였다.     
     
- 이미지를 히스토그램 평활화한 후에 같은 원리로 반복문을 돌리면 히스토그램 평활화를 영상에도 적용할 수 있다는 것을 알게되었다.     
      
- 병렬 프로그래밍을 통해 GPU를 이용하여 보다 빠른 속도로 알고리즘을 수행하는 것을 목표로 프로젝트를 진행하였다.
     
- CPU와 GPU의 성능 차이를 알아보기 위해 CPU로 돌렸을 때와 GPU로 돌렸을 때의 실행 속도 차이를 계산하여 출력하였다.

>CPU 이미지 histogram equalization 소요시간 = 1611ms   
CPU 영상 histogram equalization 소요시간 = 40140ms

>GPU 영상 histogram equalization 소요시간 = 10347ms

영상 링크 - https://www.youtube.com/watch?v=ND2n_fHTnLU
