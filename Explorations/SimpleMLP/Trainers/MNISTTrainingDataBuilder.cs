using Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Common.DataStructures;

namespace SimpleMLP.Trainers
{
    public class MNISTTrainingDataBuilder : ITrainingDataBuilder
    {
        public List<TrainingData> TrainingData { get; private set; }

        public List<TrainingData> ValidationData { get; private set; }

        public List<TrainingData> TestData { get; private set; }

        public void BuildTrainingData()
        {
            const int sizeOfValidationData = 10000;

            List<TrainingData> trainingData = BuildTrainingDataFromMNIST(
                "train-labels.idx1-ubyte", "train-images.idx3-ubyte");
            List<TrainingData> validationData = new List<TrainingData>();

            for (int c = 0; c < sizeOfValidationData; c++)
            {
                validationData.Add(trainingData[0]);
                trainingData.RemoveAt(0);
            }

            TrainingData = trainingData;

            ValidationData = validationData;

            TestData = BuildTrainingDataFromMNIST(
                "t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte");
        }

        private bool EquivalentOutputs(double[] y1, double[] y2)
        {
            bool equal = true;

            for (int c = 0; c < y1.Length; c++)
            {
                double y1Val = y1[c];
                double y2Val = y2[c];

                if ((y1Val >= .5 && y2Val < .5) ||
                    (y1Val < .5 && y2Val >= .5))
                {
                    equal = false;
                    break;
                }
            }

            return equal;
        }

        public double GradeResults(ITrainedNeuralNetwork network, List<TrainingData> testData)
        {
            int numberCorrect = 0;
            foreach (TrainingData t in testData)
            {
                double[] outputs = network.Execute(t.X);

                if (EquivalentOutputs(outputs, t.Y))
                {
                    numberCorrect++;
                }
            }

            return numberCorrect / (testData.Count * 1.0);
        }

        private static List<TrainingData> BuildTrainingDataFromMNIST(string labelsFile, string imagesFile)
        {
            List<TrainingData> ret = new List<TrainingData>();

            List<MNIST.DigitImage> images = MNIST.Reader.Read(labelsFile, imagesFile);

            foreach (MNIST.DigitImage image in images)
            {
                int height = image.pixels.Length;
                int width = image.pixels[0].Length;

                double[] imagePixels = new double[width * height];

                for (int y = 0, i = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++, i++)
                    {
                        imagePixels[i] = image.pixels[x][y] / 255.0;
                    }
                }

                double[] label = new double[10];
                label[image.label] = 1;

                ret.Add(new TrainingData
                {
                    X = imagePixels,
                    Y = label,
                    XWidth = width,
                    XHeight = height,
                    Label = image.label,
                });
            }

            return ret;
        }
    }
}
