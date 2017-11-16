using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Math;

namespace Math
{
    public static class Sigmoid
    {
        public static double Compute(double x)
        {
            return 1.0 / (1.0 + Exp(-x));
        }

        public static double ComputeDerivative(double x)
        {
            double sigmoid = Compute(x);
            return sigmoid * (1 - sigmoid);
        }
    }
}
