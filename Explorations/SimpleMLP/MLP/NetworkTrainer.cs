using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
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
            double stepSize, int numberOfEpochs, int batchSize)
        {
            int trainingDataLength = trainingData.Count;
            int reportCount = 0;

            Random random = new Random();
            for (int epochs = 0; epochs < numberOfEpochs; epochs++)
            {
                Batch[] batches = Batch.CreateBatches(trainingData, batchSize, random);

                for (int b = 0; b < batches.Length; b++)
                {
                    Batch batch = batches[b];
                    double debugAverageOutputError = 0.0;
                    // iterate over each instance of the training data. 
                    for (int n = 0; n < batch.Size; n++)
                    {
                        // get an instance of training data. 
                        TrainingData data = batch.Data[n];

                        // set the input to the net. 
                        network.SetInputLayer(data.X);
                        // feed forward. 
                        double[] result = network.Feedforward();
                        // back propagation
                        debugAverageOutputError += network.Backpropagation(data.Y);
                    };

                    if (reportCount % 1000 == 0)
                    {
                        double averaged = (debugAverageOutputError / (trainingDataLength * 1.0));
                        //fout.WriteLine(debugAverageOutputError / (trainingDataLength * 1.0)); 
                        Console.WriteLine($"Epoch {epochs}, Batch {b}, network error: {averaged * 1000} (scaled)");

                        File.AppendAllText("C:/users/brush/desktop/averageError.csv", averaged.ToString() + "\n"); 
                    }
                    reportCount++;

                    // update the network. 
                    network.UpdateNetwork(stepSize);
                }
            }
        }

        public double Test(Network network, List<TrainingData> testingData)
        {
            int numberCorrect = 0;
            foreach (TrainingData testData in testingData)
            {
                double[] outputs = network.Execute(testData.X);
                //int classFromOutputs = ClassFromOutputs(outputs);

                //string path = Path.Combine($"C:/users/brush/desktop/images/{classFromOutputs}/{Guid.NewGuid().ToString()}.bmp");
                //string directory = Path.GetDirectoryName(path); 

                //if(!Directory.Exists(directory))
                //{
                //    Directory.CreateDirectory(directory); 
                //}

                //WriteTrainingDataToDisk(testData, path); 

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
