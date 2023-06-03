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
        private double[,,] weights;
        private double[,] nodes;
        private double[,] bias;

        //Construct n-Layer Deep NN with random weights/bias
        public NeuralNetwork(int layers, int width)
        {
            Random rand = new Random();
            weights = new double[layers - 1, width, width];
            nodes = new double[layers, width];
            bias = new double[layers, width];
        }

        private double sigmoid(double z)
        {
            return 1 / 1 + Math.Pow(Math.E, -z);
        }
    }
}
