﻿using System;
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
                            int magic1 = brImages.ReadInt32(); // discard
                            int numImages = brImages.ReadInt32();
                            int numRows = brImages.ReadInt32();
                            int numCols = brImages.ReadInt32();

                            int magic2 = brLabels.ReadInt32();
                            int numLabels = brLabels.ReadInt32();

                            byte[][] pixels = new byte[28][];
                            for (int i = 0; i < pixels.Length; ++i)
                                pixels[i] = new byte[28];

                            // each test image
                            for (int di = 0; di < 10000; ++di)
                            {
                                for (int i = 0; i < 28; ++i)
                                {
                                    for (int j = 0; j < 28; ++j)
                                    {
                                        byte b = brImages.ReadByte();
                                        pixels[i][j] = b;
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
