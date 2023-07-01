using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNistNN
{
    internal class DenseLayerNetwork
    {
        //NN components for neurons
        private int layers;
        private const double learnRate = 0.01;
        private List<double[,]> weights;
        private List<double[]> activation;
        private List<double[]> bias;
        private double[] inputError;

        //Construct n-Layer Deep NN with random weights/bias
        public DenseLayerNetwork(int[] structure)
        {
            Random rand = new Random();
            weights = new List<double[,]>();
            activation = new List<double[]>();
            bias = new List<double[]>();
            layers = structure.Length;

            //Node Bias initialisation
            foreach (int layerWidth in structure)
            {
                activation.Add(new double[layerWidth]);
                bias.Add(new double[layerWidth]);
            }

            //Weight initialisation
            for(int i = 0; i < layers - 1; i++)
            {
                weights.Add(new 
                double[structure[i + 1], structure[i]]);
            }

            //Bias Randomisation
            for(int x = 0; x < layers; x++)
            {
                for(int y = 0; y < bias[x].Length; y++)
                {
                    bias[x][y] = rand.NextDouble() * 0.1;
                    if(rand.NextDouble() < 0.5)
                    {
                        bias[x][y] *= -1;
                    }
                }
            }

            //Weights Randomisation
            for(int x = 0; x < layers - 1; x++)
            {
                for(int y = 0; y < weights[x].GetLength(0); y++)
                {
                    for(int i = 0; i < weights[x].GetLength(1); i++)
                    {
                        weights[x][y, i] = rand.NextDouble() * 0.1;
                        if(rand.NextDouble() < 0.5)
                        {
                            weights[x][y, i] *= -1;
                        }
                    }
                }
            }

            inputError = new double[activation[0].Length];
        }

        //Activation function
        private double A(double z)
        {
            return Math.Max(0, z);
            return 1 / (1 + Math.Pow(Math.E, z * -1));
        }

        //Derivative of activation function
        private double dA(double a)
        {
            //return a * (1 - a);
            if(a > 0) { return 1; }
            return 0;
        }

        public void SetInput(byte[,] image)
        {
            for(int x = 0; x < 28; x++)
            {
                for (int y = 0; y < 28; y++)
                {
                    //Data normalisation
                    if (image[x, y] == Convert.ToByte(255))
                    {
                        activation[0][(x + 1) * (y + 1) - 1] = 0;
                    }
                    else
                    {
                        activation[0][(x + 1) * (y + 1) - 1] =
                        (255 - Convert.ToDouble(image[x, y])) / 255;
                    }
                }
            }
            for(int x = 1; x < layers; x++)
            {
                for(int y = 0; y < activation[x].Length; y++)
                {
                    activation[x][y] = 0;
                }
            }
        }

        //Feed forward
        public void Run()
        {
            for(int i = 0; i < layers - 1; i++)
            {
                for(int x = 0; x < activation[i + 1].Length; x++)
                {
                    double z = 0;
                    for(int y = 0; y < activation[i].Length; y++)
                    {
                        z += weights[i][x, y] * activation[i][y];
                    }
                    z += bias[i + 1][x];
                    activation[i + 1][x] = A(z);
                }
            }
        }

        //Backpropagate
        public void Backpropagate(double[] expected, List<double[]> errorBias, List<double[,]>errorWeight)
        {
            List<double[]> errorBtmp = new List<double[]>();

            foreach (double[] layer in activation)
            {
                errorBtmp.Add(new double[layer.Length]);
            }

            //Compute errors in output for bias
            for(int x = 0; x < activation[layers - 1].Length; x++)
            {
                errorBtmp[layers - 1][x] = (activation[layers - 1][x] - expected[x]) *
                dA(activation[layers - 1][x]);
                errorBias[layers - 1][x] += errorBtmp[layers - 1][x];
            }

            //Compute errors in output for weights
            for(int x = 0; x < activation[layers - 2].Length; x++)
            {
                for(int y = 0; y < activation[layers - 1].Length; y++)
                {
                    errorWeight[layers - 2][y, x] += errorBtmp[layers - 1][y] * 
                    activation[layers - 2][x];
                }
            }

            //Backpropagate errors, deriving partial derivatives of cost function
            for(int i = layers - 2; i > 0; i--)
            {
                for(int x = 0; x < activation[i].Length; x++)
                {
                    double sum = 0;
                    for(int y = 0; y < activation[i + 1].Length; y++)
                    {
                        sum += weights[i][y, x] * errorBtmp[i + 1][y] *
                        dA(activation[i][x]);
                    }
                    errorBtmp[i][x] = sum;
                    errorBias[i][x] += sum;
                }
                for(int x = 0; x < activation[i - 1].Length; x++)
                {
                    for(int y = 0; y < activation[i].Length; y++)
                    {
                        errorWeight[i - 1][y, x] += errorBtmp[i][y] * 
                        activation[i - 1][x];
                    }
                }
            }

            //Computed for potential prior convolution layers
            for (int x = 0; x < activation[0].Length; x++)
            {
                double sum = 0;
                for (int y = 0; y < activation[1].Length; y++)
                {
                    sum += weights[0][y, x] * errorBtmp[1][y] *
                    dA(activation[0][x]);
                }
                errorBtmp[0][x] = sum;
                inputError[x] += sum;
            }
        }

        //Perform weight/bias adjustment via gradient descent
        public void GradientDescent(List<byte[,]> images, byte[] labels)
        {
            double[] expected;
            List<double[]> errorBias = new List<double[]>();
            List<double[,]> errorWeight = new List<double[,]>();

            foreach (double[] layer in activation)
            {
                errorBias.Add(new double[layer.Length]);
            }

            for (int x = 0; x < layers - 1; x++)
            {
                errorWeight.Add(new double[activation[x + 1].Length,
                activation[x].Length]);
            }

            for (int x = 0; x < labels.Length; x++)
            {
                SetInput(images[x]);
                Run();

                expected = new double[10];
                expected[Convert.ToInt32(labels[x])] = 1;

                Backpropagate(expected, errorBias, errorWeight);
            }

            //Bias adjustment
            for (int x = 1; x < layers; x++)
            {
                for(int y = 0; y < bias[x].Length; y++)
                {
                    bias[x][y] -= learnRate 
                    * (errorBias[x][y] / labels.Length);
                }
            }

            //Weights adjustment
            for(int i = 0; i < layers - 1; i++)
            {
                for(int x = 0; x < activation[i + 1].Length; x++)
                {
                    for(int y = 0; y < activation[i].Length; y++)
                    {
                        weights[i][x, y] -= learnRate * 
                        (errorWeight[i][x, y] / labels.Length);
                    }
                }
            }

            //Average input errors
            for(int x = 0; x < inputError.Length; x++)
            {
                inputError[x] /= labels.Length;
            }
        }

        public double[] GetResult()
        {
            double max = 0;
            int value = 0;
            for(int x = 0; x < activation[layers - 1].Length; x++)
            {
                if (activation[layers - 1][x] > max)
                {
                    max = activation[layers - 1][x];
                    value = x;
                }
            }
            double[] result = { value, max };
            return result;
        }

        public double[] GetInputError()
        {
            return inputError;
        }

    }
}
