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

        //Assumed convolutional-pooling pattern for structure
        //NOTE: kernel sets must be same depth as input 
        public ConvolutionLayerNetwork(int iX, int iY, int iZ, int[,] structure)
        {
            //Network structure initialisation
            layers = structure.GetLength(0) + 1;
            kernels = new List<double[,,,]>();
            activation = new List<double[,,]>();
            bias = new List<double[,,]>();

            activation.Add(new double[iX, iY, iZ]);
            bias.Add(new double[iX, iY, iZ]);

            for(int i = 0; i < structure.GetLength(0); i++)
            {
                kernels.Add(new double[structure[i, 0],
                iZ, structure[i, 1], structure[i, 2]]);

                iX -= structure[i, 0] - 1;
                iY -= structure[i, 1] - 1;
                iZ = structure[i, 2];

                activation.Add(new double[iX, iY, iZ]);

                bias.Add(new double[iX, iY, iZ]);
            }

            //Bias Randomisation
            randomiseBias();

            //Kernel Randomisation
            randomiseKernel();
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
