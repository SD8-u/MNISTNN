using MNistNN;
// See https://aka.ms/new-console-template for more information

Console.WriteLine("MNIST------NEURAL NETWORK-----------");

int[] structure = { 784, 20, 20, 10 };
NeuralNetwork network = new NeuralNetwork(structure);
List<byte[,]> images = new List<byte[,]>();

byte[] labels = MNISTReader.ReadLabels(60000);
images = MNISTReader.ReadImages(60000);

for(int e = 0; e < 10; e++)
{
    for (int i = 0; i < 50000; i += 1)
    {
        byte[] testLabels = new byte[1];
        for (int x = i; x < i + 1; x++)
        {
            testLabels[x - i] = labels[x];
        }
        network.GradientDescent(images.GetRange(i, 1), testLabels);
        if (i == 0) { Console.WriteLine("EPOCH: " + e); network.Display(); Console.WriteLine("ACTUAL: " + testLabels[0]); }
    }
}

for(int x = 50000; x < 50010; x++)
{
    network.SetInput(images[x]);
    network.Run();
    network.Display();
    Console.WriteLine("ACTUAL: " + labels[x]);
}

/*byte[,] im = new byte[28, 28];
for(int x = 0; x < 28; x++)
{
    for(int y = 0; y < 28; y++)
    {
        im[x, y] = Convert.ToByte(255);
    }
}
network.SetInput(im);
network.Run();
network.Display();*/

