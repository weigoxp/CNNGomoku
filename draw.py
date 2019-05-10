import matplotlib.pylab as plt



def draw(accuracies):

    training = accuracies[0]
    #validation = [x/250 for x in  set[1] ]
    testing =  accuracies[1]

    ks =[ i for i in range(len(training))]
    f1 = plt.figure(1)
    plt.title('accuracy')
  #  plt.xticks(ks)
    plt.ylim([0, 1])
    plt.plot(ks,training, color='red',label='training',alpha=0.5)
  #  plt.plot(ks,validation, color='blue',label='validation',alpha=0.5)
    plt.plot(ks,testing, color='green',label='testing',alpha=0.5)

    plt.legend(loc='upper right')
    f1.show()