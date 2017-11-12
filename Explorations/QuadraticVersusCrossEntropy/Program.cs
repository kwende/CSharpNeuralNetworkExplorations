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
        static double[] SingleNeuronQuadratic(
            double initialWeight, double initialBias, int numberOfIterations, double stepSize)
        {
            double bias = initialBias;
            double weight = initialWeight;

            double[] ret = new double[numberOfIterations];
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
                ret[c] = activation;
            }

            return ret;
        }

        static double[] SingleNeuronCrossEntropy(
            double initialWeight, double initialBias, int numberOfIterations, double stepSize)
        {
            double bias = initialBias;
            double weight = initialWeight;

            double[] ret = new double[numberOfIterations];
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
                ret[c] = activation;
            }

            return ret;
        }

        static void Main(string[] args)
        {
            double[] quadratic = SingleNeuronQuadratic(2.0, 2.0, 1000, .5);
            double[] crossEntropy = SingleNeuronCrossEntropy(2.0, 2.0, 1000, .5);

            using (StreamWriter sw = File.CreateText("comparison.csv"))
            {
                sw.WriteLine("Quadratic Cost, Cross Entropy Cost");
                for (int c = 0; c < quadratic.Length; c++)
                {
                    sw.WriteLine($"{quadratic[c]},{crossEntropy[c]}"); 
                }
            }
        }
    }
}
