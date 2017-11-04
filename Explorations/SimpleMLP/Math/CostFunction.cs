using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Math;

namespace SimpleMLP.Math
{
    public static class CostFunction
    {
        public static double Compute(double target, double output)
        {
            double diff = output - target;
            return (1 / 2.0) * Pow(diff, 2);
        }

        public static double ComputeDerivative(double target, double output)
        {
            return target - output;
        }
    }
}
