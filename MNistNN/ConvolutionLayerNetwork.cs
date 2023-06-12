using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNistNN
{
    internal class ConvolutionLayerNetwork
    {
        int layers;
        List<double[,,,]> kernels; 
        List<double[,,]> activation;
        List<double[,,]> bias;
        bool[] convPoolConfig;

        //NOTE: kernel sets must be same depth as input 
        public ConvolutionLayerNetwork(int iX, int iY, int iZ, int[,] structure, 
        double[,,] input, bool[] convPoolConfig)
        {
            //Network structure initialisation
            layers = structure.GetLength(0) + 1;
            kernels = new List<double[,,,]>();
            activation = new List<double[,,]>();
            bias = new List<double[,,]>();
            this.convPoolConfig = convPoolConfig;

            activation.Add(new double[iX, iY, iZ]);
            bias.Add(new double[iX, iY, iZ]);

            activation[0] = input;

            for (int i = 0; i < structure.GetLength(0); i++)
            {
                kernels.Add(new double[structure[i, 0],
                iZ, structure[i, 1], structure[i, 2]]);

                iX -= (structure[i, 0] - 1);
                iY -= (structure[i, 1] - 1);
                iZ = structure[i, 2];

                activation.Add(new double[iX, iY, iZ]);

                bias.Add(new double[iX, iY, iZ]);
            }

            //Bias Randomisation
            randomiseBias();

            //Kernel Randomisation
            randomiseKernel();
            this.convPoolConfig = convPoolConfig;
        }

        //Feedforward
        public void Run()
        {
            for(int i = 1; i < layers; i++)
            {
                //Convolutional layer
                if (convPoolConfig[i - 1])
                {
                    //Compute for each kernel set
                    for (int k = 0; k < kernels[i - 1].GetLength(0); k++)
                    {
                        //Convolve and sum for each depth of the input
                        for (int x = 0; x < activation[i - 1].GetLength(0); x++)
                        {
                            double[,,] conv = convolution(k, x,
                            kernels[i - 1], activation[i - 1]);
                            sum(k, 0, activation[i], conv);
                        }
                        //Apply bias and activation to layer
                        sum(k, k, activation[i], bias[i], true);
                    }
                }
                //Pooling layer
                else
                {
                    for(int a = 0; a < activation[i - 1].GetLength(0); a++)
                    {
                        maxPool(i, a);
                    }
                }
            }
        }

        //Max pooling function
        private void maxPool(int i, int a)
        {
            int k1 = kernels[i - 1].GetLength(2);
            int k2 = kernels[i - 1].GetLength(3);
            for (int b = 0; b < activation[i - 1].GetLength(1)
            - (k1 - 1); b += k1 + 1)
            {
                for (int c = 0; c < activation[i - 1].GetLength(2)
                - (k2 - 1); c += k2 + 1)
                {
                    double max = Double.MinValue;
                    for (int x = b; x < b + kernels[i - 1].GetLength(2); x++)
                    {
                        for (int y = c; y < c + kernels[i - 1].GetLength(3); y++)
                        {
                            max = Math.Max(max, activation[i - 1][a, x, y]);
                        }
                    }
                    activation[i][a, b / (k1 + 1), c / (k2 + 1)] = max;
                }
            }
        }

        //Activation function
        private double A(double z)
        {
            return Math.Max(0, z);
            return 1 / (1 + Math.Pow(Math.E, z * -1));
        }

        private double[,,] convolution(int a, int b, double[,,,] m1, double[,,] m2)
        {
            int k1 = (m1.GetLength(2) - m2.GetLength(1)) + 1;
            int k2 = (m1.GetLength(3) - m2.GetLength(2)) + 1;
            double[,,] result = new double[1, k1, k2];

            for(int x = 0; x < k1; x++)
            {
                for(int y = 0; y < k2; y++)
                {
                    double sum = 0;
                    for(int z = k1; z < k1 + m2.GetLength(1); z++)
                    {
                        for(int w = k2; w < k2 + m2.GetLength(2); w++)
                        {
                            sum += m1[a, b, z, w] * m2[b, z - k1, w - k2];
                        }
                    }
                    result[0, x, y] = sum;
                }
            }

            return result;
        }

        private void sum(int a, int b, double[,,] m1, double[,,] m2, bool activ=false)
        {
            for(int x = 0; x < m1.GetLength(1); x++)
            {
                for(int y = 0; y < m1.GetLength(2); y++)
                {
                    m1[a, x, y] += m2[b, x, y];
                    if (activ)
                    {
                        m1[a, x, y] = A(m1[a, x, y]);
                    }
                }
            }
        }

        private void randomiseBias()
        {
            Random rand = new Random();
            foreach (double[,,] layer in bias)
            {
                for (int x = 0; x < layer.GetLength(0); x++)
                {
                    for (int y = 0; y < layer.GetLength(1); y++)
                    {
                        for (int z = 0; z < layer.GetLength(2); z++)
                        {
                            layer[x, y, z] = rand.NextDouble() * 0.1;
                            if (rand.NextDouble() < 0.5)
                            {
                                layer[x, y, z] *= 1;
                            }
                        }
                    }
                }
            }
        }

        private void randomiseKernel()
        {
            Random rand = new Random();
            foreach (double[,,,] layer in kernels)
            {
                for (int x = 0; x < layer.GetLength(0); x++)
                {
                    for (int y = 0; y < layer.GetLength(1); y++)
                    {
                        for (int z = 0; z < layer.GetLength(2); z++)
                        {
                            for (int w = 0; w < layer.GetLength(3); w++)
                            {
                                layer[x, y, z, w] = rand.NextDouble() * 0.1;
                                if (rand.NextDouble() < 0.5)
                                {
                                    layer[x, y, z, w] *= -1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
