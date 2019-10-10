from matplotlib import pyplot as plt

def plot():
    with open("cost_history.txt","r+") as f:
        raw_data = f.read().strip().split(' ')
    cost = [float(i) for i in raw_data]
    iter = [i for i in range(0, len(raw_data))]
    plt.scatter(iter, cost, s=0.3,alpha = 0.8)
    plt.xlabel('Iteration step')
    plt.ylabel('Cost')
    plt.title('Cost-Iteration step Figure')
    plt.show()

if __name__ == "__main__":
    plot()