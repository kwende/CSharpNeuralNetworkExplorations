using Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Math;

namespace Math.CostFunctions
{
    public class CrossEntropyCostFunction : ICostFunction
    {
        public double Compute(double target, double output)
        {
            return (output * Log(output)) * ((1 - target) * Log(1 - output));
        }

        public double ComputeDerivativeWRTActivation(double output, double target)
        {
            return (output - target) / (output * (1 - output));
        }
    }
}
