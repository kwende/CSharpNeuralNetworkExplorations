using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace QuadraticVersusCrossEntropy
{
    class Program
    {
        static void SingleNeuronQuadratic(
            double initialWeight, double initialBias, int numberOfIterations, double stepSize, string outputFile)
        {
            double bias = initialBias;
            double weight = initialWeight;

            using (StreamWriter sw = File.CreateText(outputFile))
            {
                double input = 1;
                for (int c = 0; c < numberOfIterations; c++)
                {
                    double result = weight * input + bias;
                    double activation = Math.Sigmoid.Compute(result);

                    double delta = Math.MeanSquaredErrorCostFunction.ComputeDerivativeWRTActivation(activation, 0.0) *
                        Math.Sigmoid.ComputeDerivative(result);

                    bias = bias - stepSize * delta;
                    weight = weight - stepSize * (delta * input);

                    input = activation;

                    sw.WriteLine(activation);
                    Console.WriteLine(activation);
                }
            }
        }

        static void SingleNeuronCrossEntropy(
            double initialWeight, double initialBias, int numberOfIterations, double stepSize, string outputFile)
        {
            double bias = initialBias;
            double weight = initialWeight;

            using (StreamWriter sw = File.CreateText(outputFile))
            {
                double input = 1;
                for (int c = 0; c < numberOfIterations; c++)
                {
                    double result = weight * input + bias;
                    double activation = Math.Sigmoid.Compute(result);

                    double delta = Math.CrossEntropyCostFunction.ComputeDerivativeWRTActivation(activation, 0.0) *
                        Math.Sigmoid.ComputeDerivative(result);

                    bias = bias - stepSize * delta;
                    weight = weight - stepSize * (delta * input);

                    input = activation;

                    sw.WriteLine(activation);
                    Console.WriteLine(activation);
                }
            }
        }

        static void Main(string[] args)
        {
            SingleNeuronQuadratic(2.0, 2.0, 1000, .5, "quadratic_cost_bad.csv");
            SingleNeuronCrossEntropy(2.0, 2.0, 1000, .5, "cross_entropy_good.csv");
        }
    }
}
