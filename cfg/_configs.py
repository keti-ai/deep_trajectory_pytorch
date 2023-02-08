import argparse

class Parser ():
    def __init__(self):
# 인자값을 받을 수 있는 인스턴스 생성
        parser = argparse.ArgumentParser(description='Argparse Tutorial')

        # 입력받을 인자값 설정 (default 값 설정가능)
        parser.add_argument('--epoch',          type=int,   default=150)
        parser.add_argument('--batch_size',     type=int,   default=128)
        parser.add_argument('--lr_initial',     type=float, default=0.1)

        # args 에 위의 내용 저장
        self.args    = parser.parse_args()

        # 입력받은 인자값 출력
        print(self.args.epoch)
        print(self.args.batch_size)
        print(self.args.lr_initial)

