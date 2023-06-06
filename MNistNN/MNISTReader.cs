using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNistNN
{
    internal static class MNISTReader
    {
        private const string imagePath = "C:/Users/7sava/Desktop/Programs/Other/MNistNN/" +
                "Dataset/train-images.idx3-ubyte";
        private const string labelPath = "C:/Users/7sava/Desktop/Programs/Other/MNistNN/" +
                "Dataset/train-labels.idx1-ubyte";
        public static List<byte[,]> ReadImages(int n)
        {
            BinaryReader rImages = new BinaryReader(new FileStream(imagePath, FileMode.Open));

            int m1 = rImages.ReadBigInt32();
            int nImages = rImages.ReadBigInt32();
            int nRows = rImages.ReadBigInt32();
            int nCols = rImages.ReadBigInt32();

            List<byte[,]> images = new List<byte[,]>();
            for(int i = 0; i < n; i++)
            {
                images.Add(new byte[28, 28]);
                for(int x = 0; x < 28; x++)
                {
                    for(int y = 0; y < 28; y++)
                    {
                        images[i][x, y] = rImages.ReadByte();
                    }
                }
            }
            rImages.Close();
            return images;
        }

        public static byte[] ReadLabels(int n)
        {
            BinaryReader rlabels = new BinaryReader(new FileStream(labelPath, FileMode.Open));

            int m1 = rlabels.ReadBigInt32();
            int nlabels = rlabels.ReadBigInt32();

            byte[] labels = new byte[n];

            for(int i = 0; i < n; i++)
            {
                labels[i] = rlabels.ReadByte();
            }

            rlabels.Close();
            return labels;
        }
    }

    public static class Extensions
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
    }
}
