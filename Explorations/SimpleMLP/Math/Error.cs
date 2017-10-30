using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.Math
{
    public static class Error
    {
        public static double ComputeErrorForOutputNeuron(double expectedOutput, double actualOutput, int totalNumberOfTrainingExamples)
        {
            return (Math.CostFunction.ComputeDerivative(actualOutput, expectedOutput) *
                    Math.Sigmoid.ComputeDerivative(actualOutput)) / (totalNumberOfTrainingExamples * 1.0);
        }
    }
}
