using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNistNN
{
    internal class NeuralNetwork
    {
        //NN components for neurons
        private int layers;
        private List<double[,]> weights;
        private List<double[]> activation;
        private List<double[]> bias;
        private List<double[]> nodes;

        //Construct n-Layer Deep NN with random weights/bias
        public NeuralNetwork(int[] structure)
        {
            Random rand = new Random();
            weights = new List<double[,]>();
            activation = new List<double[]>();
            bias = new List<double[]>();
            nodes = new List<double[]>();
            layers = structure.Length;

            //Node Bias initialisation
            foreach(int layerWidth in structure)
            {
                activation.Add(new double[layerWidth]);
                nodes.Add(new double[layerWidth]);
                bias.Add(new double[layerWidth]);
            }

            //Weight initialisation
            for(int i = 0; i < layers - 1; i++)
            {
                weights.Add(new 
                double[structure[i], structure[i + 1]]);
            }

            //Bias Randomisation
            for(int x = 0; x < layers; x++)
            {
                for(int y = 0; y < bias[x].Length; y++)
                {
                    bias[x][y] = rand.NextDouble() * rand.Next(5);
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
                        weights[x][y, i] = rand.NextDouble() * rand.Next(10);
                        if(rand.NextDouble() < 0.5)
                        {
                            weights[x][y, i] *= -1;
                        }
                    }
                }
            }
        }

        //Sigmoid - Activation function
        private double sigmoid(double z)
        {
            return 1 / 1 + Math.Pow(Math.E, -z);
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
                        z += weights[i][x, y] * activation[i][x];
                    }
                    z += bias[i + 1][x];
                    nodes[i + 1][x] = z;
                    activation[i + 1][x] = sigmoid(z);
                }
            }
        }

        //Backpropagate
        public List<double[]> Backpropagate(double[] expected)
        {
            List<double[]> error = new List<double[]>();
            foreach (double[] layer in activation)
            {
                error.Add(new double[layer.Length]);
            }

            //Compute errors in output
            for(int x = 0; x < activation[layers - 1].Length; x++)
            {
                error[layers - 1][x] = (activation[layers - 1][x] - expected[x]) *
                (activation[layers - 1][x] * (1 - activation[layers - 1][x]));
            }

            //Backpropagate errors, deriving partial derivatives of cost
            for(int i = layers - 2; i > 0; i--)
            {
                for(int x = 0; x < activation[i].Length; x++)
                {
                    double sum = 0;
                    for(int y = 0; y < activation[i + 1].Length; y++)
                    {
                        sum += weights[i][y, x] * error[i + 1][y] *
                        (activation[i][x] * (1 - activation[i][x]));
                    }
                    error[i][x] = sum;
                }
            }
            return error;
        }
    }
}
