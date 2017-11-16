using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST
{
    //https://jamesmccaffrey.wordpress.com/2013/11/23/reading-the-mnist-data-set-with-c/

    public class Reader
    {
        private static int SwapEndian(int input)
        {
            int ret = 0x0;

            ret |= input >> 24;
            ret |= (input & 0x00FF0000) >> 8;
            ret |= (input & 0x0000FF00) << 8;
            ret |= (input & 0x000000FF) << 24; 

            return ret; 
        }

        public static List<DigitImage> Read(string labelsFile, string imagesFile)
        {
            List<DigitImage> digitImages = new List<DigitImage>();

            using (FileStream ifsLabels = new FileStream(labelsFile, FileMode.Open))
            {
                using (FileStream ifsImages = new FileStream(imagesFile, FileMode.Open)) // test images
                {
                    using (BinaryReader brLabels = new BinaryReader(ifsLabels))
                    {
                        using (BinaryReader brImages = new BinaryReader(ifsImages))
                        {
                            int magic1 = SwapEndian(brImages.ReadInt32()); // discard
                            int numImages = SwapEndian(brImages.ReadInt32());
                            int numRows = SwapEndian(brImages.ReadInt32());
                            int numCols = SwapEndian(brImages.ReadInt32());

                            int magic2 = SwapEndian(brLabels.ReadInt32());
                            int numLabels = SwapEndian(brLabels.ReadInt32());

                            byte[][] pixels = new byte[numCols][];
                            for (int i = 0; i < pixels.Length; ++i)
                                pixels[i] = new byte[numRows];

                            // each test image
                            for (int di = 0; di < numImages; ++di)
                            {
                                for (int i = 0; i < numRows; ++i)
                                {
                                    for (int j = 0; j < numCols; ++j)
                                    {
                                        byte b = brImages.ReadByte();
                                        pixels[j][i] = b;
                                    }
                                }

                                byte lbl = brLabels.ReadByte();

                                DigitImage dImage =
                                  new DigitImage(pixels, lbl);

                                digitImages.Add(dImage);
                            } // each image
                        }
                    }
                }
            }
            return digitImages;
        }
    }
}
