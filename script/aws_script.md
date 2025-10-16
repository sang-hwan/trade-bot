```bash
# EC2 접속 -> 접속 키 & EC2 IPv4 주소
ssh -i "trade-bot-key.pem" ubuntu@13.239.243.132

# 서버 최신 상태 업데이트
sudo apt update && sudo apt upgrade -y

# 서버 재시작
sudo reboot

# python 및 pip 설치
sudo apt install python3-pip -y

# 코드 가져오기
git clone https://github.com/sang-hwan/trade-bot.git

# 가져온 코드로 이동하기
cd trade-bot

# 가상환경 도구 설치
sudo apt install python3.12-venv -y

# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 필요한 라이브러리 설치하기
pip3 install -r requirements.txt

# EC2 종료(정지)
sudo shutdown -h now
```