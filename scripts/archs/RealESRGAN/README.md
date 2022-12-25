# RealESRGAN Paper Review
![](https://images.velog.io/images/heaseo/post/000c6fe2-227b-4ff3-930f-83a09fec26b3/Real-ESRGAN%20degradation.png)

Blind super-resolution을 통해 복잡하고 알 수 없는 열화가 있는 저해상도 이미지를 복원하려는 수 차례 시도가 많았지만 완벽하게 복원하기는 힘들었었다. 이 논문은 기존의 강력한 ESRGAN super-resolution 모델을 바탕으로 다양한 전처리 열화기법 단계를 추가해서 Real-ESRGAN 모델로 확장한다. 실생활에서 생기는 열화를 구현하기 위해 high-order 열화기법이 도입됬다. 게다가, 논문의 연구진은 rining과 overshoot 현상도 고려해서 새로운 열화 전처리 단계를 만들었다. 추가적으로, U-Net discriminator에 spectral normalization를 추가해서 판별능력을 올려 안정적인 학습을 할 수 있도록 도왔다. 결과적으로, 광범위한 실제 데이터셋을 이용한 비교를 통해 Real-ESRGAN이 기존 ESRGAN보다 시각적으로 뛰어난 성능을 입증했다. [더보기](https://heaseochung.github.io/RealESRGAN.html)
