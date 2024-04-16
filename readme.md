## 선진 돈사 재고 관리 AICCTV

- 도커 컴포즈 설치
. 해당 시스템을 이용하기 위해선 docker-compose를 이용해야 합니다. <br>
. 시스템의 숫자가 늘어나도, docker-compose를 통해 일괄 배포합니다. <br>
. 시스템 수정 사항이 있을 시, 해당 레포지토리만 수정하여 일괄 수정, 일괄 배포가 가능하도록 구성합니다. <br>

```
sudo apt install -y python3 python3-pip
sudo pip3 install docker-compose
sudo pip3 install -U docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose -version # 설치 확인
```

- 도커 컴포즈 설치 시, python request 에러
. 설치 시, 아래 에러가 발생할 수 있습니다.

```
ERROR: docker 7.0.0 has requirement requests>=2.26.0, but you'll have requests 2.22.0 which is incompatible.
```

. 이 때, 아래와 같이 대응해주세요.

```
pip install --upgrade requests
```

- 도커 컴포즈 설치 시, unexpected keyword argument 'ssl_version'에러 해결

```
pip install docker==6.1.3
```