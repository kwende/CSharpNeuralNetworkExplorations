using Common.DataStructures;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLP
{
    public class NetworkTrainer
    {
        static void WriteTrainingDataToDisk(TrainingData data, string outputFile)
        {
            using (Bitmap bmp = new Bitmap(data.XWidth, data.XHeight))
            {
                for (int y = 0, i = 0; y < data.XHeight; y++)
                {
                    for (int x = 0; x < data.XWidth; x++, i++)
                    {
                        byte v = (byte)(data.X[i] * 255);
                        bmp.SetPixel(x, y, Color.FromArgb(v, v, v));
                    }
                }

                bmp.Save(outputFile);
            }
        }

        public void Train(Network network, List<TrainingData> trainingData,
            double stepSize, int numberOfEpochs,
            int batchSize, ValidationDataOptions validationOptions,
            Action<LearningProgress> onLearningProgress,
            Action<double> onValidationDataReport)
        {
            int trainingDataLength = trainingData.Count;

            int counter = 0;
            for (int epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                network.UpdateDroputLayers();

                Batch[] batches = Batch.CreateBatches(trainingData, batchSize, network.NetworkRandom);

                for (int b = 0; b < batches.Length; b++)
                {
                    Batch batch = batches[b];
                    double outputError = 0.0;
                    // iterate over each instance of the training data. 
                    for (int n = 0; n < batch.Size; n++)
                    {
                        // get an instance of training data. 
                        TrainingData data = batch.Data[n];
                        // feed forward. 
                        double[] result = network.Feedforward(data.X);
                        // back propagation
                        outputError += network.Backpropagation(data.Y);
                    };

                    if (onLearningProgress != null)
                    {
                        onLearningProgress(new LearningProgress
                        {
                            BatchNumber = b,
                            Epoch = epoch,
                            CurrentNetworkError = outputError,
                            Counter = ++counter,
                        });
                    }
                    // update the network. 
                    network.UpdateNetwork(stepSize, trainingDataLength, batch.Size);
                }

                if (validationOptions != null && onValidationDataReport != null)
                {
                    if ((epoch + 1) % validationOptions.NumberOfEpochsBetweenTests == 0)
                    {
                        double accuracy = Test(network, validationOptions.ValidationData);
                        onValidationDataReport(accuracy);
                    }
                }
            }
        }

        public double Test(Network network, List<TrainingData> testingData)
        {
            int numberCorrect = 0;
            foreach (TrainingData testData in testingData)
            {
                double[] outputs = network.Execute(testData.X);

                int classFromOutputs = ClassFromOutputs(outputs);

                //if (!Directory.Exists($"C:/users/brush/desktop/Groups/{classFromOutputs}"))
                //{
                //    Directory.CreateDirectory($"C:/users/brush/desktop/Groups/{classFromOutputs}");
                //}

                //WriteTrainingDataToDisk(testData, $"C:/users/brush/desktop/Groups/{classFromOutputs}/{Guid.NewGuid().ToString().Replace("-", "")}.bmp"); 

                if (EquivalentOutputs(outputs, testData.Y))
                {
                    numberCorrect++;
                }
            }

            return numberCorrect / (testingData.Count * 1.0);
        }

        private int ClassFromOutputs(double[] outputs)
        {
            int maxIndex = 0;
            double maxValue = double.MinValue;

            for (int c = 0; c < outputs.Length; c++)
            {
                if (outputs[c] > maxValue)
                {
                    maxValue = outputs[c];
                    maxIndex = c;
                }
            }

            return maxIndex;
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
    }
}
