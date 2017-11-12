using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Math;

namespace Math
{
    public static class CrossEntropyCostFunction
    {
        public static double Compute(double target, double output)
        {
            return (output * Log(output)) * ((1 - target) * Log(1 - output));
        }

        public static double ComputeDerivativeWRTActivation(double output, double target)
        {
            return (output - target) / (output * (1 - output));
        }
    }
}
