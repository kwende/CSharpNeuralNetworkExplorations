using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Math;

namespace Math
{
    public static class MeanSquaredErrorCostFunction
    {
        public static double Compute(double target, double output)
        {
            double diff = output - target;
            return (1 / 2.0) * Pow(diff, 2);
        }

        public static double ComputeDerivativeWRTActivation(double output, double target)
        {
            return output - target;
        }
    }
}
