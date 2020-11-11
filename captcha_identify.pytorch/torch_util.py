import torch
import matplotlib.pyplot as plt

def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(force_cpu=False):
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    if not cuda:
        print('Using CPU')
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        print("Using CUDA device0 _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
              (x[0].name, x[0].total_memory / c))
        if ng > 0:
            # torch.cuda.set_device(0)  # OPTIONAL: Set GPU ID
            for i in range(1, ng):
                print("           device%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                      (i, x[i].name, x[i].total_memory / c))
    print('')  # skip a line
    return device

def plot_result():
    fig = plt.figure(15)
    with open("results.txt", "r") as f:
        import csv
        f_csv = csv.reader(f)
        epoch , acc = [], []
        for row in f_csv:
            print(row)
            epoch.append(int(row[0]))
            acc.append(float(row[1]))
        plt.plot(epoch, acc)
    plt.title("result-epoch&acc") 
    plt.xlabel("epoch") 
    plt.ylabel("accuracy") 
    plt.show()
    fig.savefig("results.png", dpi=300)

if __name__=="__main__":
    plot_result()