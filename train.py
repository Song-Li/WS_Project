from src.sklearn import SKLearn

def run():
    sklearn = SKLearn("./data/aidata.log", 'rf')
    data, label = sklearn.pre_data(True)
    sklearn.train(data, label)

run()
