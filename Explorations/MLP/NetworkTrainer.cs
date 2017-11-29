using Common.DataStructures;
using Common.Interfaces;
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
        //static void WriteTrainingDataToDisk(TrainingData data, string outputFile)
        //{
        //    using (Bitmap bmp = new Bitmap(data.XWidth, data.XHeight))
        //    {
        //        for (int y = 0, i = 0; y < data.XHeight; y++)
        //        {
        //            for (int x = 0; x < data.XWidth; x++, i++)
        //            {
        //                byte v = (byte)(data.X[i] * 255);
        //                bmp.SetPixel(x, y, Color.FromArgb(v, v, v));
        //            }
        //        }

        //        bmp.Save(outputFile);
        //    }
        //}

        public void Train(Network network, ITrainingDataBuilder trainingDataBuilder,
            double stepSize, int numberOfEpochs,
            int batchSize, int numberOfEpochsBeforeValidating,
            Action<LearningProgress> onLearningProgress,
            Action<double> onValidationDataReport)
        {
            int trainingDataLength = trainingDataBuilder.TrainingData.Count;

            int counter = 0;
            for (int epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                network.UpdateDroputLayers();

                Batch[] batches = Batch.CreateBatches(trainingDataBuilder.TrainingData, batchSize, network.NetworkRandom);

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

                if (trainingDataBuilder.ValidationData != null)
                {
                    if ((epoch + 1) % numberOfEpochsBeforeValidating == 0)
                    {
                        double accuracy = trainingDataBuilder.GradeResults(network, trainingDataBuilder.ValidationData);
                        onValidationDataReport(accuracy);
                    }
                }
            }
        }

        //public double Test(Network network, List<TrainingData> testingData)
        //{

        //}

        //private int ClassFromOutputs(double[] outputs)
        //{
        //    int maxIndex = 0;
        //    double maxValue = double.MinValue;

        //    for (int c = 0; c < outputs.Length; c++)
        //    {
        //        if (outputs[c] > maxValue)
        //        {
        //            maxValue = outputs[c];
        //            maxIndex = c;
        //        }
        //    }

        //    return maxIndex;
        //}
    }
}
