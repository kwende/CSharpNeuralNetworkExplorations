using Common.DataStructures;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Interfaces
{
    public interface ITrainingDataBuilder
    {
        void BuildTrainingData();
        double GradeResults(ITrainedNeuralNetwork network, List<TrainingData> testData);
        List<TrainingData> TrainingData { get; }
        List<TrainingData> ValidationData { get; }
        List<TrainingData> TestData { get; }
    }
}
