from src.sklearn import SKLearn
import argparse

def run(data_path, algo):
    print(f"Selected algorithm {algo}")
    sklearn = SKLearn(data_path, algo)
    data, label = sklearn.pre_data(pca=False)
    sklearn.train(data, label, over_sampling=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Training and testing')
    parser.add_argument('-d', '--data-path', type=str, help="Set the path to the data file")
    parser.add_argument('-a', '--algorithm', type=str, 
            help="Set the algorithm for training, can be rf, mlp")
    return parser.parse_args() 

if __name__ == "__main__":
    args = parse_args()
    data_path = './data/aidata.log'
    algo = 'rf'
    if args.data_path is not None:
        data_path = args.data_path
    if args.algorithm is not None:
        algo = args.algorithm

    run(data_path, algo)
