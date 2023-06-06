using MNistNN;
// See https://aka.ms/new-console-template for more information

Console.WriteLine("MNIST------NEURAL NETWORK-----------");

int[] structure = { 784, 20, 20, 10 };
NeuralNetwork network = new NeuralNetwork(structure);
List<byte[,]> images = new List<byte[,]>();

byte[] labels = MNISTReader.ReadLabels(100);
images = MNISTReader.ReadImages(100);

for(int i = 0; i < 100; i++)
{
    Console.WriteLine(Convert.ToInt32(labels[i]));
}

network.GradientDescent(images, labels);

