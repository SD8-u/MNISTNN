using MNistNN;

Console.WriteLine("MNIST------NEURAL NETWORK-----------");

//Construct Network
int[] structure = { 784, 100, 10 };
NeuralNetwork network = new NeuralNetwork(structure);

//Read Dataset
byte[] labels = MNISTReader.ReadLabels(60000);
List<byte[,]> images = MNISTReader.ReadImages(60000);

//Perform Training
int batchSize = 1;
int trainSize = 50000;
int testSize = 10000;
int epochs = 20;

for (int e = 0; e < epochs; e++)
{
    for (int i = 0; i < trainSize; i += batchSize)
    {
        byte[] testLabels = new byte[batchSize];
        for (int x = i; x < i + batchSize; x++)
        {
            testLabels[x - i] = labels[x];
        }
        network.GradientDescent(images.GetRange(i, batchSize), testLabels);
        if (i >= trainSize - batchSize) 
        { 
            //Perform epoch test
            Console.WriteLine("EPOCH: " + e);
            double correct = 0;
            for (int x = trainSize; x < trainSize + testSize; x++)
            {
                network.SetInput(images[x]);
                network.Run();
                if (network.GetResult()[0] == labels[x])
                {
                    correct++;
                }
            }
            Console.WriteLine("ACCURACY: " + correct / testSize * 100 + "%");
        }
    }
}


