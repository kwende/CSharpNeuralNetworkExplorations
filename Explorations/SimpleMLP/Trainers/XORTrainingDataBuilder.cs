using Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Common.DataStructures;

namespace SimpleMLP.Trainers
{
    public class XORTrainingDataBuilder : ITrainingDataBuilder
    {
        public List<TrainingData> TrainingData { get; private set; }

        public List<TrainingData> ValidationData { get; private set; }

        public List<TrainingData> TestData { get; private set; }

        public void BuildTrainingData()
        {
            List<TrainingData> trainingData = new List<TrainingData>();

            trainingData.Add(new TrainingData
            {
                X = new double[2] { 0, 0 },
                Y = new double[1] { 0 },
            });

            trainingData.Add(new TrainingData
            {
                X = new double[2] { 0, 1 },
                Y = new double[1] { 1 },
            });

            trainingData.Add(new TrainingData
            {
                X = new double[2] { 1, 0 },
                Y = new double[1] { 1 },
            });

            trainingData.Add(new TrainingData
            {
                X = new double[2] { 1, 1 },
                Y = new double[1] { 0 },
            });

            TrainingData = trainingData;
            ValidationData = null;
            TestData = trainingData;
        }

        public double GradeResults(ITrainedNeuralNetwork network, List<TrainingData> dataToGrade)
        {
            int correct = 0;
            foreach (TrainingData testDataInstance in dataToGrade)
            {
                double[] result = network.Execute(testDataInstance.X);
                if ((result[0] >= .5 && testDataInstance.Y[0] == 1) ||
                    (result[0] < .5 && testDataInstance.Y[0] == 0))
                {
                    correct++;
                }
            }

            return (correct / (dataToGrade.Count * 1.0));
        }
    }
}
