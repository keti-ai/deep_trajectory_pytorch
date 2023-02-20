import argparse

class Parser ():
    def __init__(self):
# 인자값을 받을 수 있는 인스턴스 생성
        parser = argparse.ArgumentParser(description='deep_Trajextory_code')


        # model parameters
        parser.add_argument('--model_type', type=str, default='cnn', help='Type of model architecture')
        parser.add_argument('--hidden_dim', type=int, default=512, help='Dimension of hidden layers')
        parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the model')

        # data parameters
        parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing training data')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')

        # training parameters
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
        parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
        parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory for saving model checkpoints')

        # args 에 위의 내용 저장
        self.args    = parser.parse_args()

        # 입력받은 인자값 출력
        p_len = 100
        p_1="model parpameters"
        print("#"*(p_len//2-len(p_1)//2),p_1,"#"*(p_len//2-len(p_1)//2))
        print("model_type : ",self.args.model_type)
        print("hidden_dim : ",self.args.hidden_dim)
        print("num_layers : ",self.args.num_layers)
        p_2="data parpameters"
        print("#"*(p_len//2-len(p_2)//2),p_2,"#"*(p_len//2-len(p_2)//2))
        print("data_dir : ",self.args.data_dir)
        print("batch_size : ",self.args.batch_size)
        print("num_workers : ",self.args.num_workers)
        p_3 = "training parpameters"
        print("#" * (p_len // 2 - len(p_3) // 2), p_3, "#" * (p_len // 2 - len(p_3) // 2))
        print("learning_rate : ",self.args.learning_rate)
        print("num_epochs : ",self.args.num_epochs)
        print("save_dir : ",self.args.save_dir)
        print("#"*p_len)


